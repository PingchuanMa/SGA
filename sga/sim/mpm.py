from typing import Optional, Any, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from torch import Tensor
import warp as wp
from warp.context import Devicelike

import sga
from sga.warp import Tape, CondTape
from .base import Statics, State, Model, ModelBuilder, StateInitializer, StaticsInitializer, ShapeLike


@wp.struct
class MPMStatics(Statics):

    vol: wp.array(dtype=float)
    rho: wp.array(dtype=float)
    clip_bound: wp.array(dtype=float)

    def init(self, shape: ShapeLike, device: Devicelike = None) -> None:
        self.vol = wp.zeros(shape=shape, dtype=float, device=device, requires_grad=False)
        self.rho = wp.zeros(shape=shape, dtype=float, device=device, requires_grad=False)
        self.clip_bound = wp.zeros(shape=shape, dtype=float, device=device, requires_grad=False)

    def update_vol(self, sections: list[int], vols: list[float]) -> None:
        offset = 0
        for section, vol in zip(sections, vols):
            wp.launch(self.set_float, dim=self.vol.shape, inputs=[self.vol, offset, offset + section, vol], device=self.vol.device)
            offset += section

    def update_rho(self, sections: list[int], rhos: list[float]) -> None:
        offset = 0
        for section, rho in zip(sections, rhos):
            wp.launch(self.set_float, dim=self.rho.shape, inputs=[self.rho, offset, offset + section, rho], device=self.rho.device)
            offset += section

    def update_clip_bound(self, sections: list[int], clip_bounds: list[float]) -> None:
        offset = 0
        for section, clip_bound in zip(sections, clip_bounds):
            wp.launch(self.set_float, dim=self.clip_bound.shape, inputs=[
                self.clip_bound, offset, offset + section, clip_bound], device=self.clip_bound.device)
            offset += section


@wp.struct
class MPMParticleData(object):

    x: wp.array(dtype=wp.vec3)
    v: wp.array(dtype=wp.vec3)
    C: wp.array(dtype=wp.mat33)
    stress: wp.array(dtype=wp.mat33)

    F: wp.array(dtype=wp.mat33)

    @staticmethod
    @wp.kernel
    def init_F_kernel(F: wp.array(dtype=wp.mat33)) -> None:
        p = wp.tid()
        F[p] = wp.identity(n=3, dtype=float)

    def init_F(self) -> None:
        wp.launch(self.init_F_kernel, dim=self.F.shape, inputs=[self.F], device=self.F.device)

    def init(self, shape: ShapeLike, device: Devicelike = None, requires_grad: bool = False) -> None:

        self.x = wp.zeros(shape=shape, dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self.v = wp.zeros(shape=shape, dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self.C = wp.zeros(shape=shape, dtype=wp.mat33, device=device, requires_grad=requires_grad)
        self.stress = wp.zeros(shape=shape, dtype=wp.mat33, device=device, requires_grad=requires_grad)

        self.F = wp.empty(shape=shape, dtype=wp.mat33, device=device, requires_grad=requires_grad)
        self.init_F()

    def clear(self) -> None:
        self.x.zero_()
        self.v.zero_()
        self.C.zero_()
        self.stress.zero_()

        self.init_F()

    def zero_grad(self) -> None:
        if self.x.requires_grad:
            self.x.grad.zero_()
        if self.v.requires_grad:
            self.v.grad.zero_()
        if self.C.requires_grad:
            self.C.grad.zero_()
        if self.F.requires_grad:
            self.F.grad.zero_()
        if self.stress.requires_grad:
            self.stress.grad.zero_()

    def clone(self, requires_grad: Optional[bool] = None) -> 'MPMParticleData':
        clone = MPMParticleData()
        clone.x = wp.clone(self.x, requires_grad=requires_grad)
        clone.v = wp.clone(self.v, requires_grad=requires_grad)
        clone.C = wp.clone(self.C, requires_grad=requires_grad)
        clone.F = wp.clone(self.F, requires_grad=requires_grad)
        clone.stress = wp.clone(self.stress, requires_grad=requires_grad)
        return clone

@wp.struct
class MPMGridData(object):

    v: wp.array(dtype=wp.vec3, ndim=3)
    mv: wp.array(dtype=wp.vec3, ndim=3)
    m: wp.array(dtype=float, ndim=3)

    def init(self, shape: ShapeLike, device: Devicelike = None, requires_grad: bool = False) -> None:

        self.v = wp.zeros(shape=shape, dtype=wp.vec3, ndim=3, device=device, requires_grad=requires_grad)
        self.mv = wp.zeros(shape=shape, dtype=wp.vec3, ndim=3, device=device, requires_grad=requires_grad)
        self.m = wp.zeros(shape=shape, dtype=float, ndim=3, device=device, requires_grad=requires_grad)

    def clear(self) -> None:
        self.v.zero_()
        self.mv.zero_()
        self.m.zero_()

    def zero_grad(self) -> None:
        if self.v.requires_grad:
            self.v.grad.zero_()
        if self.mv.requires_grad:
            self.mv.grad.zero_()
        if self.m.requires_grad:
            self.m.grad.zero_()

    def clone(self, requires_grad: Optional[bool] = None) -> 'MPMGridData':
        clone = MPMGridData()
        clone.v = wp.clone(self.v, requires_grad=requires_grad)
        clone.mv = wp.clone(self.mv, requires_grad=requires_grad)
        clone.m = wp.clone(self.m, requires_grad=requires_grad)
        return clone


@wp.struct
class MPMConstant(object):

    num_grids: int
    dt: float
    bound: int
    gravity: wp.vec3
    dx: float
    inv_dx: float
    eps: float


class MPMState(State):

    def __init__(self, shape: int, device: Devicelike = None, requires_grad: bool = False) -> None:

        super().__init__(shape, device, requires_grad)

        particle = MPMParticleData()
        particle.init(shape, device, requires_grad)
        self.particle = particle

    def zero_grad(self) -> None:
        self.particle.zero_grad()

    def clear(self) -> None:
        self.particle.clear()

    def clone(self, requires_grad: Optional[bool] = None) -> 'MPMState':
        clone = MPMState(self.shape, self.device, requires_grad)
        clone.particle = self.particle.clone(requires_grad)
        return clone

    def to_torch(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        def convert(x):
            return wp.to_torch(x, requires_grad=False).requires_grad_(x.requires_grad)

        x = convert(self.particle.x)
        v = convert(self.particle.v)
        C = convert(self.particle.C)
        F = convert(self.particle.F)
        stress = convert(self.particle.stress)
        return x, v, C, F, stress

    def to_torch_grad(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        def convert(x):
            return wp.to_torch(x.grad, requires_grad=False) if x.grad is not None else None

        grad_x = convert(self.particle.x)
        grad_v = convert(self.particle.v)
        grad_C = convert(self.particle.C)
        grad_F = convert(self.particle.F)
        grad_stress = convert(self.particle.stress)
        return grad_x, grad_v, grad_C, grad_F, grad_stress

    def from_torch(
            self,
            x: Optional[Tensor] = None,
            v: Optional[Tensor] = None,
            C: Optional[Tensor] = None,
            F: Optional[Tensor] = None,
            stress: Optional[Tensor] = None) -> None:

        def convert(x, dtype):
            array = wp.from_torch(x.contiguous(), dtype, requires_grad=False)
            array.requires_grad = x.requires_grad
            return array

        if x is not None:
            self.particle.x = convert(x, wp.vec3)
        if v is not None:
            self.particle.v = convert(v, wp.vec3)
        if C is not None:
            self.particle.C = convert(C, wp.mat33)
        if F is not None:
            self.particle.F = convert(F, wp.mat33)
        if stress is not None:
            self.particle.stress = convert(stress, wp.mat33)

    def from_torch_grad(
            self,
            grad_x: Optional[Tensor] = None,
            grad_v: Optional[Tensor] = None,
            grad_C: Optional[Tensor] = None,
            grad_F: Optional[Tensor] = None,
            grad_stress: Optional[Tensor] = None) -> None:
        """
        Warp uses array.grad to store gradients, so we need array.grad.assign(tensor.grad)
        instead of array.grad = tensor.grad. I miss c++ so much.
        """

        def convert(a, t):
            a.grad.assign(wp.from_torch(t.contiguous(), a.dtype, requires_grad=False))

        if grad_x is not None:
            convert(self.particle.x, grad_x)
        if grad_v is not None:
            convert(self.particle.v, grad_v)
        if grad_C is not None:
            convert(self.particle.C, grad_C)
        if grad_F is not None:
            convert(self.particle.F, grad_F)
        if grad_stress is not None:
            convert(self.particle.stress, grad_stress)


class MPMModel(Model):

    ConstantType = MPMConstant
    StaticsType = MPMStatics
    StateType = MPMState

    def __init__(self, constant: ConstantType, device: Devicelike = None, requires_grad: bool = False) -> None:
        super().__init__(constant, device)
        self.requires_grad = requires_grad
        self.grid_op = None

        shape = (self.constant.num_grids, self.constant.num_grids, self.constant.num_grids)
        grid = MPMGridData()
        grid.init(shape, device, requires_grad)
        self.grid = grid

    def forward(
            self,
            statics: MPMStatics,
            state_curr: MPMState,
            state_next: MPMState,
            tape: Optional[Tape] = None) -> None:

        device = self.device
        constant = self.constant
        particle_curr = state_curr.particle
        particle_next = state_next.particle
        grid = self.grid

        num_grids = constant.num_grids
        num_particles = particle_curr.x.shape[0]

        grid.clear()
        grid.zero_grad()

        wp.launch(self.p2g, dim=num_particles, inputs=[constant, statics, particle_curr, grid], device=device)
        wp.launch(self.grid_op, dim=[num_grids] * 3, inputs=[constant, grid], device=device)

        with CondTape(tape, self.requires_grad):
            wp.launch(self.g2p, dim=num_particles, inputs=[constant, statics, particle_curr, particle_next, grid], device=device)

    def backward(self, statics: MPMStatics, state_curr: MPMState, state_next: MPMState, tape: Tape) -> None:

        device = self.device
        constant = self.constant
        particle_curr = state_curr.particle
        grid = self.grid

        num_grids = constant.num_grids
        num_particles = particle_curr.x.shape[0]

        grid.clear()
        grid.zero_grad()

        local_tape = Tape()
        with local_tape:
            wp.launch(self.p2g, dim=num_particles, inputs=[constant, statics, particle_curr, grid], device=device)
            wp.launch(self.grid_op, dim=[num_grids] * 3, inputs=[constant, grid], device=device)

        tape.backward()

        local_tape.backward()

    @staticmethod
    @wp.kernel
    def p2g(
            constant: ConstantType,
            statics: StaticsType,
            particle_curr: MPMParticleData,
            grid: MPMGridData) -> None:

        p = wp.tid()

        p_mass = statics.vol[p] * statics.rho[p]

        p_x = particle_curr.x[p] * constant.inv_dx
        base_x = int(p_x[0] - 0.5)
        base_y = int(p_x[1] - 0.5)
        base_z = int(p_x[2] - 0.5)
        f_x = p_x - wp.vec3(
            float(base_x),
            float(base_y),
            float(base_z))

        # quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx,fx-1,fx-2]
        wa = wp.vec3(1.5) - f_x
        wb = f_x - wp.vec3(1.0)
        wc = f_x - wp.vec3(0.5)

        # wp.mat33(col_vec, col_vec, col_vec)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.75) - wp.cw_mul(wb, wb),
            wp.cw_mul(wc, wc) * 0.5,
        )

        stress = (-constant.dt * statics.vol[p] * 4.0 * constant.inv_dx * constant.inv_dx) * particle_curr.stress[p]
        affine = stress + p_mass * particle_curr.C[p]

        for i in range(3):
            for j in range(3):
                for k in range(3):

                    offset = wp.vec3(float(i), float(j), float(k))
                    dpos = (offset - f_x) * constant.dx
                    weight = w[0, i] * w[1, j] * w[2, k]
                    mv = weight * (p_mass * particle_curr.v[p] + affine * dpos)
                    m = weight * p_mass

                    wp.atomic_add(grid.mv, base_x + i, base_y + j, base_z + k, mv)
                    wp.atomic_add(grid.m, base_x + i, base_y + j, base_z + k, m)

    @staticmethod
    @wp.kernel
    def grid_op_freeslip(
            constant: ConstantType,
            grid: MPMGridData) -> None:

        px, py, pz = wp.tid()

        v = wp.vec3(0.0)
        if grid.m[px, py, pz] > 0.0:
            v = grid.mv[px, py, pz] / (grid.m[px, py, pz] + constant.eps) + constant.gravity * constant.dt
        else:
            v = constant.gravity * constant.dt

        if px < constant.bound and v[0] < 0.0:
            v = wp.vec3(0.0, v[1], v[2])
        if py < constant.bound and v[1] < 0.0:
            v = wp.vec3(v[0], 0.0, v[2])
        if pz < constant.bound and v[2] < 0.0:
            v = wp.vec3(v[0], v[1], 0.0)
        if px > constant.num_grids - constant.bound and v[0] > 0.0:
            v = wp.vec3(0.0, v[1], v[2])
        if py > constant.num_grids - constant.bound and v[1] > 0.0:
            v = wp.vec3(v[0], 0.0, v[2])
        if pz > constant.num_grids - constant.bound and v[2] > 0.0:
            v = wp.vec3(v[0], v[1], 0.0)

        grid.v[px, py, pz] = v

    @staticmethod
    @wp.kernel
    def grid_op_noslip(
            constant: ConstantType,
            grid: MPMGridData) -> None:

        px, py, pz = wp.tid()

        v = wp.vec3(0.0)
        if grid.m[px, py, pz] > 0.0:
            v = grid.mv[px, py, pz] / (grid.m[px, py, pz] + constant.eps) + constant.gravity * constant.dt
        else:
            v = constant.gravity * constant.dt

        if px < constant.bound and v[0] < 0.0:
            v = wp.vec3(0.0)
        if py < constant.bound and v[1] < 0.0:
            v = wp.vec3(0.0)
        if pz < constant.bound and v[2] < 0.0:
            v = wp.vec3(0.0)
        if px > constant.num_grids - constant.bound and v[0] > 0.0:
            v = wp.vec3(0.0)
        if py > constant.num_grids - constant.bound and v[1] > 0.0:
            v = wp.vec3(0.0)
        if pz > constant.num_grids - constant.bound and v[2] > 0.0:
            v = wp.vec3(0.0)

        grid.v[px, py, pz] = v

    @staticmethod
    @wp.kernel
    def g2p(
            constant: ConstantType,
            statics: StaticsType,
            particle_curr: MPMParticleData,
            particle_next: MPMParticleData,
            grid: MPMGridData) -> None:

        p = wp.tid()

        p_x = particle_curr.x[p] * constant.inv_dx
        base_x = int(p_x[0] - 0.5)
        base_y = int(p_x[1] - 0.5)
        base_z = int(p_x[2] - 0.5)
        f_x = p_x - wp.vec3(
            float(base_x),
            float(base_y),
            float(base_z))

        # quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx,fx-1,fx-2]
        wa = wp.vec3(1.5) - f_x
        wb = f_x - wp.vec3(1.0)
        wc = f_x - wp.vec3(0.5)

        # wp.mat33(col_vec, col_vec, col_vec)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.75) - wp.cw_mul(wb, wb),
            wp.cw_mul(wc, wc) * 0.5,
        )

        new_v = wp.vec3(0.0)
        new_C = wp.mat33(0.0)

        for i in range(3):
            for j in range(3):
                for k in range(3):

                    offset = wp.vec3(float(i), float(j), float(k))
                    dpos = (offset - f_x) * constant.dx
                    weight = w[0, i] * w[1, j] * w[2, k]

                    v = grid.v[base_x + i, base_y + j, base_z + k]
                    new_v = new_v + weight * v
                    new_C = new_C + (4.0 * weight * constant.inv_dx * constant.inv_dx) * wp.outer(v, dpos)

        particle_next.v[p] = new_v
        particle_next.C[p] = new_C
        particle_next.F[p] = (wp.identity(n=3, dtype=float) + constant.dt * new_C) * particle_curr.F[p]

        bound = statics.clip_bound[p] * constant.dx
        new_x = particle_curr.x[p] + constant.dt * new_v
        new_x = wp.vec3(
            wp.clamp(new_x[0], 0.0 + bound, 1.0 - bound),
            wp.clamp(new_x[1], 0.0 + bound, 1.0 - bound),
            wp.clamp(new_x[2], 0.0 + bound, 1.0 - bound),
        )
        particle_next.x[p] = new_x


class MPMModelBuilder(ModelBuilder):

    StateType = MPMState
    ConstantType = MPMConstant
    ModelType = MPMModel

    def parse_cfg(self, cfg: sga.config.physics.sim.BaseSimConfig) -> 'MPMModelBuilder':

        num_grids: int = cfg['num_grids']
        dt: float = cfg['dt']
        bound: int = cfg['bound']
        gravity: np.ndarray = np.array(cfg['gravity'], dtype=np.float32)
        bc: str = cfg['bc']
        eps: float = cfg['eps']

        dx: float = 1 / num_grids
        inv_dx: float = float(num_grids)

        self.config['num_grids'] = num_grids
        self.config['dt'] = dt
        self.config['bound'] = bound
        self.config['gravity'] = gravity
        self.config['dx'] = dx
        self.config['inv_dx'] = inv_dx
        self.config['bc'] = bc
        self.config['eps'] = eps

        return self

    def build_constant(self) -> ConstantType:

        constant = super().build_constant()
        constant.num_grids = self.config['num_grids']
        constant.dt = self.config['dt']
        constant.bound = self.config['bound']
        constant.gravity = wp.vec3(*self.config['gravity'])
        constant.dx = self.config['dx']
        constant.inv_dx = self.config['inv_dx']
        constant.eps = self.config['eps']

        return constant

    def finalize(self, device: Devicelike = None, requires_grad: bool = False) -> ModelType:
        model = super().finalize(device, requires_grad)
        if self.config['bc'] == 'freeslip':
            model.grid_op = MPMModel.grid_op_freeslip
        elif self.config['bc'] == 'noslip':
            model.grid_op = MPMModel.grid_op_noslip
        else:
            raise ValueError('invalid boundary condition: {}'.format(self.config['bc']))
        return model


@dataclass
class MPMInitData(object):

    rho: float
    clip_bound: float

    num_particles: int
    vol: float

    pos: np.ndarray
    lin_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    center: np.ndarray = None
    ind_vel: np.ndarray = None

    def __post_init__(self) -> None:
        if self.center is None:
            self.center = self.pos.mean(0)

    @classmethod
    def get(cls, cfg: sga.config.physics.env.BaseEnvConfig) -> 'MPMInitData':
        kwargs: dict = None
        if cfg['shape']['name'].startswith('cube'):
            kwargs = cls.get_cube(
                cfg['shape']['center'],
                cfg['shape']['size'],
                cfg['shape']['resolution'],
                cfg['shape']['mode'],
            )
        elif cfg['shape']['type'] == 'mesh':
            kwargs = cls.get_mesh(
                cfg['shape']['name'],
                cfg['shape']['center'],
                cfg['shape']['size'],
                cfg['shape']['resolution'],
                cfg['shape']['mode'],
                cfg['shape']['sort']
            )
        else:
            raise ValueError('invalid shape type: {}'.format(cfg['shape']['type']))
        return cls(rho=cfg['rho'], clip_bound=cfg['clip_bound'], **kwargs)

    @classmethod
    def get_mesh(
            cls,
            name: str,
            center: list | np.ndarray,
            size: list | np.ndarray,
            resolution: int,
            mode: str,
            sort: Optional[int]) -> Mapping[str, Any]:

        center = np.array(center)
        size = np.array(size)

        asset_root = Path(__file__).resolve().parent.parent / 'assets'
        precompute_name = f'{name}_{resolution}_{mode}.npz'

        if (asset_root / precompute_name).is_file():
            file = np.load(asset_root / precompute_name)
            p_x = file['p_x']
            vol = file['vol']
        else:

            import trimesh

            mesh: trimesh.Trimesh = trimesh.load(asset_root / f'{name}.obj', force='mesh')

            # if not mesh.is_watertight:
            #     raise ValueError('invalid mesh: not watertight')

            bounds = mesh.bounds.copy()

            if mode == 'uniform':

                mesh.vertices = (mesh.vertices - bounds[0]) / (bounds[1] - bounds[0])
                dims = np.linspace(np.zeros(3), np.ones(3), resolution).T
                grid = np.stack(np.meshgrid(*dims, indexing='ij'), axis=-1).reshape(-1, 3)
                p_x = grid[mesh.contains(grid)]
                p_x = (p_x * (bounds[1] - bounds[0]) + bounds[0] - bounds.mean(0)) / (bounds[1] - bounds[0]).max()

            else:
                raise ValueError('invalid mode: {}'.format(mode))

            if sort is not None:
                indices = np.array(list(sorted(range(p_x.shape[0]), reverse=True, key=lambda x: p_x[:, sort][x])))
                p_x = p_x[indices]

            vol = mesh.volume / p_x.shape[0]
            np.savez(asset_root / precompute_name, p_x=p_x, vol=vol)

        vol = vol * np.prod(size)
        p_x = p_x * size + center
        p_x = np.ascontiguousarray(p_x.reshape(-1, 3))

        return {'num_particles': p_x.shape[0], 'vol': vol, 'pos': p_x, 'center': center}

    @classmethod
    def get_cube(
            cls,
            center: list | np.ndarray,
            size: list | np.ndarray,
            resolution: int,
            mode: str) -> dict[str, Any]:

        asset_root = Path(__file__).resolve().parent.parent / 'assets'
        size_str = '_'.join([str(s) for s in size])
        precompute_name = f'cube_{resolution}_{mode}_{size_str}.npz'

        center = np.array(center)
        size = np.array(size)

        if (asset_root / precompute_name).is_file():
            file = np.load(asset_root / precompute_name)
            p_x = file['p_x']
            vol = file['vol']
        else:
            resolutions = np.around(size * resolution / size.max()).astype(int)
            if mode == 'uniform':
                dims = [np.linspace(l, r, res) for l, r, res in zip(-size / 2, size / 2, resolutions)]
                p_x = np.stack(np.meshgrid(*dims, indexing='ij'), axis=-1).reshape(-1, 3)
            elif mode == 'random':
                rng = np.random.Generator(np.random.PCG64(0))
                p_x = rng.uniform(-size / 2, size / 2, (np.prod(resolutions), 3)).reshape(-1, 3)
            else:
                raise ValueError('invalid mode: {}'.format(mode))

            vol = np.prod(size) / p_x.shape[0]
            np.savez(asset_root / precompute_name, p_x=p_x, vol=vol)

        p_x = p_x + center
        p_x = np.ascontiguousarray(p_x.reshape(-1, 3))
        return {'num_particles': p_x.shape[0], 'vol': vol, 'pos': p_x, 'center': center}

    def set_lin_vel(self, value: list | np.ndarray) -> None:
        self.lin_vel = np.array(value)

    def zero_lin_vel(self) -> None:
        self.set_lin_vel(np.zeros_like(self.lin_vel))

    def set_ang_vel(self, value: list | np.ndarray) -> None:
        self.ang_vel = np.array(value)

    def zero_ang_vel(self) -> None:
        self.set_ang_vel(np.zeros_like(self.ang_vel))

    def set_ind_vel(self, ind_vel: np.ndarray) -> None:
        self.ind_vel = np.array(ind_vel)


class MPMStateInitializer(StateInitializer):

    StateType = MPMState
    ModelType = MPMModel

    def __init__(self, model: ModelType) -> None:
        super().__init__(model)
        self.groups: list[MPMInitData] = []

    def add_group(self, group: MPMInitData) -> None:
        self.groups.append(group)

    def finalize(self) -> tuple[StateType, list[int]]:

        pos_groups = []
        vel_groups = []
        sections = []

        for group in self.groups:
            pos = group.pos.copy()

            if group.ind_vel is None:
                lin_vel = group.lin_vel.copy()
                ang_vel = group.ang_vel.copy()
                vel = lin_vel + np.cross(ang_vel, pos - group.center)
            else:
                vel = group.ind_vel.copy()

            pos_groups.append(pos)
            vel_groups.append(vel)
            sections.append(group.num_particles)

        pos_groups = np.concatenate(pos_groups, axis=0)
        vel_groups = np.concatenate(vel_groups, axis=0)

        state_0 = super().finalize(shape=pos_groups.shape[0], requires_grad=False)

        state_0.particle.x.assign(pos_groups)
        state_0.particle.v.assign(vel_groups)

        return state_0, sections


class MPMStaticsInitializer(StaticsInitializer):

    StaticsType = MPMStatics
    ModelType = MPMModel

    def __init__(self, model: ModelType) -> None:
        super().__init__(model)
        self.groups: list[MPMInitData] = []

        self.sections: list[int] = []
        self.vols: list[float] = []
        self.rhos: list[float] = []
        self.clip_bounds: list[float] = []

    def add_group(self, group: MPMInitData) -> None:
        self.groups.append(group)

    def finalize(self) -> StaticsType:

        for group in self.groups:
            self.sections.append(group.num_particles)

            self.vols.append(group.vol)
            self.rhos.append(group.rho)
            self.clip_bounds.append(group.clip_bound)

        statics = super().finalize(shape=sum(self.sections))
        statics.update_vol(self.sections, self.vols)
        statics.update_rho(self.sections, self.rhos)
        statics.update_clip_bound(self.sections, self.clip_bounds)

        return statics

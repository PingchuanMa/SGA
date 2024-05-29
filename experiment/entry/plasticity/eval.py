import sys
import dataclasses
import random
from pathlib import Path
import argparse
import yaml

import torch
import torch.nn as nn
from tqdm import trange
import numpy as np
import warp as wp

import sga
from sga.sim import MPMModelBuilder, MPMStateInitializer, MPMStaticsInitializer, MPMInitData, MPMForwardSim, VolumeElasticity, SigmaElasticity

root: Path = sga.utils.get_root(__file__)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    args, unknown_args = parser.parse_known_args()
    cfg = sga.config.EvalConfig(path=args.path)
    cfg.update(unknown_args)

    is_dataset = cfg.is_dataset
    tpos = cfg.tpos
    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)

    wp.config.quiet = True
    wp.init()
    wp_device = wp.get_device(f'cuda:{cfg.gpu}')
    wp.ScopedTimer.enabled = False
    wp.set_module_options({'fast_math': True})

    torch_device = torch.device(f'cuda:{cfg.gpu}')
    torch.backends.cudnn.benchmark = True
    sga.warp.replace_torch_svd()
    sga.warp.replace_torch_polar()
    sga.warp.replace_torch_trace()
    sga.warp.replace_torch_cbrt()
    torch.set_default_device(torch_device)

    log_root = root / 'log'
    if Path(cfg.path).is_absolute():
        exp_root = Path(cfg.path)
    else:
        exp_root = log_root / cfg.path
    if exp_root.is_relative_to(log_root):
        exp_name = str(exp_root.relative_to(log_root))
    else:
        exp_name = str(exp_root)
    img_root = exp_root / 'img'
    vid_root = exp_root / 'vid'
    state_root = exp_root / 'state'
    sga.utils.mkdir(exp_root, overwrite=cfg.overwrite, resume=cfg.resume)
    img_root.mkdir(parents=True, exist_ok=True)
    vid_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)

    cfg_dict = dataclasses.asdict(cfg)
    with (exp_root / 'config.yaml').open('w') as f:
        yaml.safe_dump(cfg_dict, f)

    # physics

    if cfg.physics.env.physics.elasticity == 'sigma':
        elasticity: nn.Module = SigmaElasticity()
    elif cfg.physics.env.physics.elasticity == 'volume':
        elasticity: nn.Module = VolumeElasticity()
    else:
        raise ValueError(f'invalid elasticity: {cfg.physics.env.physics.elasticity}')
    elasticity.to(torch_device)
    elasticity.eval()
    elasticity.requires_grad_(False)

    full_py = Path(cfg.physics.env.physics.path).read_text('utf-8')
    full_py = full_py.format(**cfg.physics.env.physics.__dict__)
    physics_py_path = exp_root / 'physics.py'
    physics_py_path.write_text(full_py, 'utf-8')

    physics: nn.Module = sga.utils.get_class_from_path(physics_py_path, 'Physics')()

    if cfg.ckpt_path is not None:
        ckpt_path = Path(cfg.ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        physics.load_state_dict(ckpt)

    physics.to(torch_device)
    physics.requires_grad_(False)
    physics.eval()

    # warp

    model = MPMModelBuilder().parse_cfg(cfg.physics.sim).finalize(wp_device)
    state_initializer = MPMStateInitializer(model)
    statics_initializer = MPMStaticsInitializer(model)

    init_data = MPMInitData.get(cfg.physics.env)
    init_data.set_lin_vel(cfg.physics.env.vel.lin_vel)
    init_data.set_ang_vel(cfg.physics.env.vel.ang_vel)

    state_initializer.add_group(init_data)
    statics_initializer.add_group(init_data)

    state, _ = state_initializer.finalize()
    statics = statics_initializer.finalize()
    sim = MPMForwardSim(model, statics)

    x, v, C, F, stress = state.to_torch()
    state_recorder = sga.utils.StateRecorder()
    state_recorder.add_hyper(key_indices=cfg.physics.env.shape.key_indices)
    if is_dataset:
        state_recorder.add(x=x, v=v, F=F, C=C, stress=stress)
    else:
        state_recorder.add(x=x, v=v)

    with sga.renderer.BaseRenderer.candidates[cfg.physics.render.name](cfg.physics.render) as renderer:

        render_step = 0

        positions: np.ndarray = x.cpu().detach().numpy()
        renderer.add_pcd(positions, 'pcd')
        renderer.save(img_root, render_step)

        for step in trange(1, cfg.physics.sim.num_steps + 1, desc=f'[eval] {exp_name}', file=sys.stdout, position=tpos, leave=None):

            F = physics(F)
            stress = elasticity(F)
            state.from_torch(F=F, stress=stress)

            x, v, C, F = sim(state)
            if is_dataset:
                state_recorder.add(x=x, v=v, F=F, C=C, stress=stress)
            else:
                state_recorder.add(x=x, v=v)

            if step % cfg.physics.render.skip_frames == 0:
                render_step += 1
                positions: np.ndarray = x.cpu().detach().numpy()
                renderer.add_pcd(positions, 'pcd')
                renderer.save(img_root, render_step)

        state_recorder.save(state_root / 'ckpt.pt')
        renderer.make_video(img_root, vid_root / f'{cfg.physics.render.name}.mp4')


if __name__ == '__main__':
    main()

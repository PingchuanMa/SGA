import sys
import dataclasses
import random
from pathlib import Path
import argparse
import math

import yaml
from tqdm import trange, tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import warp as wp

import sga
from sga.sim import MPMModelBuilder, MPMStaticsInitializer, MPMInitData, MPMCacheDiffSim, MPMDataset

root: Path = sga.utils.get_root(__file__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--dataset_path', type=str)
    args, unknown_args = parser.parse_known_args()
    cfg = sga.config.TrainConfig(path=args.path, dataset_path=args.dataset_path)
    cfg.update(unknown_args)

    num_steps = cfg.physics.sim.num_steps
    num_epochs = cfg.optim.num_epochs
    num_teacher_steps = cfg.optim.num_teacher_steps

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
    state_root = exp_root / 'state'
    ckpt_root = exp_root / 'ckpt'
    sga.utils.mkdir(exp_root, overwrite=cfg.overwrite, resume=cfg.resume)
    state_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    cfg_dict = dataclasses.asdict(cfg)
    with (exp_root / 'config.yaml').open('w') as f:
        yaml.safe_dump(cfg_dict, f)

    writer = SummaryWriter(exp_root, purge_step=0)

    # warp

    model = MPMModelBuilder().parse_cfg(cfg.physics.sim).finalize(wp_device, requires_grad=True)
    statics_initializer = MPMStaticsInitializer(model)

    init_data = MPMInitData.get(cfg.physics.env)
    init_data.set_lin_vel(cfg.physics.env.vel.lin_vel)
    init_data.set_ang_vel(cfg.physics.env.vel.ang_vel)

    statics_initializer.add_group(init_data)
    statics = statics_initializer.finalize()

    diff_sim = MPMCacheDiffSim(model, statics, num_steps=num_steps)

    # physics

    full_py = Path(cfg.physics.env.physics.path).read_text('utf-8')
    full_py = full_py.format(**cfg.physics.env.physics.__dict__)
    physics_py_path = exp_root / 'physics.py'
    physics_py_path.write_text(full_py, 'utf-8')

    physics: nn.Module = sga.utils.get_class_from_path(physics_py_path, 'Physics')()
    physics.to(torch_device)

    parametric = len(list(physics.parameters())) > 0

    if parametric:
        if cfg.optim.optimizer == 'adam':
            optimizer = torch.optim.Adam(physics.parameters(), lr=cfg.optim.lr)
        else:
            raise ValueError(f'Unknown optimizer: {cfg.optim.optimizer}')

        if cfg.optim.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.optim.num_epochs)
        elif cfg.optim.scheduler == 'none':
            scheduler = None
        else:
            raise ValueError(f'Unknown scheduler: {cfg.optim.scheduler}')

    if Path(cfg.dataset_path).is_absolute():
        dataset_root = Path(cfg.dataset_path)
    else:
        dataset_root = log_root / cfg.dataset_path
    dataset = MPMDataset(dataset_root, torch_device)

    # training

    t = trange(num_epochs + 1, desc=f'[train] {exp_name}', file=sys.stdout, position=tpos, leave=None)
    for epoch in t:

        torch.save(physics.state_dict(), ckpt_root / f'{epoch:04d}.pt')
        parametric = len(list(physics.parameters())) > 0

        if parametric:
            for name, param in physics.named_parameters():
                writer.add_scalar(f'param/{name}', param.item(), epoch)

        loss_x = 0.0
        loss_v = 0.0

        x_gt, v_gt, C_gt, F_gt, _ = dataset[0]

        # state_recorder = sga.utils.StateRecorder()
        # state_recorder.add(x=x_gt, v=v_gt)

        physics.train()
        for step in range(num_steps):

            is_teacher = step == 0 or (num_teacher_steps > 0 and step % num_teacher_steps == 0)
            if is_teacher:
                x, v, C, F = x_gt, v_gt, C_gt, F_gt

            stress = physics(F)
            x, v, C, F = diff_sim(step, x, v, C, F, stress)
            # state_recorder.add(x=x, v=v)

            x_gt, v_gt, C_gt, F_gt, _ = dataset[step + 1]
            loss_x += sga.utils.loss_fn(x, x_gt) / num_steps * cfg.optim.alpha_position
            loss_v += sga.utils.loss_fn(v, v_gt).item() / num_steps * cfg.optim.alpha_velocity

        # state_recorder.save(state_root / f'{epoch:04d}.pt')

        loss_x_item = loss_x.item()
        loss_v_item = loss_v

        if math.isnan(loss_x_item) or math.isnan(loss_v_item):
            writer.add_scalar('loss/position', float('nan') , epoch)
            writer.add_scalar('loss/velocity', float('nan') , epoch)
            tqdm.write('loss is nan')
            break

        writer.add_scalar('loss/position', loss_x_item, epoch)
        writer.add_scalar('loss/velocity', loss_v_item, epoch)
        t.set_postfix(l_x=loss_x_item, l_v=loss_v_item)

        if epoch == num_epochs:
            t.refresh()
            break

        if not parametric:
            break

        optimizer.zero_grad()
        try:
            loss_x.backward()
            clip_grad_norm_(physics.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        except RuntimeError as e:
            tqdm.write(str(e))
            break
    t.close()

    torch.save(physics.state_dict(), ckpt_root / 'final.pt')
    writer.close()

if __name__ == '__main__':
    main()

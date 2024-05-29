from pathlib import Path
import dataclasses
import sys
import os
import argparse
import random
import math
from pprint import pprint

import yaml
from tqdm import trange, tqdm
import numpy as np
import torch

import sga
from sga.agent import ConstitutivePhysicist, Population

root = sga.utils.get_root(__file__)


def get_perf_feedback(losses: dict[str, list[float]], params: dict[str, list[float]]) -> list[str]:
    feedbacks = []

    parametric = len(params) > 0

    if parametric:
        best_idx = min(range(len(losses['position'])), key=lambda i: losses['position'][i] if not math.isnan(
            losses['position'][i]) else float('inf'))

        feedbacks.append('#### Physical parameter training curves (versus iteration)')
        feedbacks.append('')  # add a blank line
        for tag, traj in sorted(params.items()):
            msg = ', '.join([f'{loss:.2f}' for loss in traj])
            msg = f'- {tag}: [{msg}] (Best: {traj[best_idx]:.2f})'
            feedbacks.append(msg)

        feedbacks.append('')  # add a blank line

        feedbacks.append('#### Loss training curves (versus iteration)')
        feedbacks.append('')  # add a blank line
        for tag, traj in sorted(losses.items()):
            msg = ', '.join([f'{loss:.4f}' for loss in traj])
            if tag == 'position':
                tag = f'{tag} (Key loss)'
            msg = f'- {tag}: [{msg}] (Best: {traj[best_idx]:.4f})'
            feedbacks.append(msg)
    else:
        feedbacks.append('#### Evaluation loss (since it is a non-parametric model)')
        feedbacks.append('')  # add a blank line
        for tag, traj in sorted(losses.items()):
            msg = f'{traj[-1]:.4f}'
            if tag == 'position':
                tag = f'{tag} (Key loss)'
            msg = f'- {tag}: [{msg}]'
            feedbacks.append(msg)

    return feedbacks


def get_state_feedback(states: dict[str, torch.Tensor | tuple], state_size: int) -> list[str]:
    feedbacks = []

    key_indices: list[int] = list(states['key_indices'])
    key_frames: list[int] = np.linspace(0, states['x'].size(0) - 1, state_size, dtype=int).tolist()

    particle_frame_pos: np.ndarray = states['x'].permute(1, 0, 2).detach().cpu().numpy().copy()
    particle_frame_vel: np.ndarray = states['v'].permute(1, 0, 2).detach().cpu().numpy().copy()

    feedbacks.append('#### Representative particle trajectories (versus time)')
    feedbacks.append('')  # add a blank line
    for i_particle, particle in enumerate(key_indices):
        feedbacks.append(f'- Particle {i_particle}')
        pos_msgs = []
        vel_msgs = []
        for frame in key_frames:
            pos_msg = ', '.join([f'{pos:.2f}' for pos in particle_frame_pos[particle, frame]])
            pos_msg = f'({pos_msg})'
            pos_msgs.append(pos_msg)
            vel_msg = ', '.join([f'{vel:.2f}' for vel in particle_frame_vel[particle, frame]])
            vel_msg = f'({vel_msg})'
            vel_msgs.append(vel_msg)
        pos_msg = ', '.join(pos_msgs)
        pos_msg = f'    - positions: [{pos_msg}]'
        feedbacks.append(pos_msg)
        vel_msg = ', '.join(vel_msgs)
        vel_msg = f'    - velocities: [{vel_msg}]'
        feedbacks.append(vel_msg)

    return feedbacks


@torch.no_grad()
def main():
    python_path = Path.home() / 'miniconda3' / 'envs' / 'sga' / 'bin' / 'python'
    my_env = os.environ.copy()

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--dataset_path', type=str)
    args, unknown_args = parser.parse_known_args()
    cfg = sga.config.DefaultConfig(path=args.path, dataset_path=args.dataset_path)
    cfg.update(unknown_args)
    pprint(cfg)

    entry = cfg.llm.entry
    train_py_path = root / 'entry' / entry / 'train.py'
    eval_py_path = root / 'entry' / entry / 'eval.py'
    base_train_cmds = [python_path, train_py_path]
    base_eval_cmds = [python_path, eval_py_path]

    tpos = cfg.tpos
    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    log_root = root / 'log'
    if Path(cfg.path).is_absolute():
        exp_root = Path(cfg.path)
    else:
        exp_root = log_root / cfg.path
    if exp_root.is_relative_to(log_root):
        exp_name = str(exp_root.relative_to(log_root))
    else:
        exp_name = str(exp_root)
    if Path(cfg.dataset_path).is_absolute():
        dataset_root = Path(cfg.dataset_path)
    else:
        dataset_root = log_root / cfg.dataset_path

    primitive_root = exp_root / 'primitive'
    offspring_root = exp_root / 'offspring'
    iteration_root = exp_root / 'iteration'
    sga.utils.mkdir(exp_root, overwrite=cfg.overwrite, resume=cfg.resume, verbose=True)
    primitive_root.mkdir(parents=True, exist_ok=True)
    offspring_root.mkdir(parents=True, exist_ok=True)
    iteration_root.mkdir(parents=True, exist_ok=True)

    cfg_dict = dataclasses.asdict(cfg)
    yaml.safe_dump(cfg_dict, (exp_root / 'config.yaml').open('w'))


    if cfg.llm.name.startswith('openai-gpt-4'):
        dataset_states = torch.load(dataset_root / 'state' / 'ckpt.pt', map_location='cpu')
        dataset_feedback = '\n'.join(get_state_feedback(dataset_states, cfg.llm.state_size))
    else:
        dataset_feedback = ''
    physicist = ConstitutivePhysicist(cfg.llm, seed=seed, env_info=dataset_feedback)
    population = Population(cfg.llm)

    for i_ind, ind_physics in enumerate(tqdm(cfg.llm.primitives, desc=f'[primitive] {exp_name}', file=sys.stdout, position=tpos)):
        ind_root = primitive_root / f'{i_ind:04d}'
        ind_root.mkdir(parents=True, exist_ok=True)

        # train
        train_args = {
            'tpos': tpos + 1,
            'path': ind_root,
            'dataset_path': cfg.dataset_path,
            'physics.env.physics': ind_physics
        }
        error = sga.utils.run_exp(base_train_cmds, train_args, unknown_args, my_env)

        losses, params = sga.utils.parse_tensorboard(ind_root)
        states = None
        fitness = min(losses['position'], key=lambda x: x if not math.isnan(x) else float('inf'))

        feedbacks = []
        if cfg.llm.name.startswith('openai-gpt-4'):
            # evaluate and render
            for eval_key in ['final']:
                eval_args = {
                    'is_dataset': False,
                    'tpos': tpos + 1,
                    'path': ind_root / 'eval' / eval_key,
                    'physics.env.physics': ind_physics,
                    'ckpt_path': ind_root / 'ckpt' / f'{eval_key}.pt'
                }
                error = sga.utils.run_exp(base_eval_cmds, eval_args, unknown_args, my_env)
            states = torch.load(ind_root / 'eval' / eval_key / 'state' / 'ckpt.pt', map_location='cpu')
            feedbacks += get_state_feedback(states, cfg.llm.state_size)
            feedbacks.append('')  # add a blank line
        feedbacks += get_perf_feedback(losses, params)
        feedback = '\n'.join(feedbacks)
        code_path = ind_root / 'physics.py'

        population.add_primitive(code_path, feedback, fitness, losses, params, states, ind_root)

    for i_iter in trange(cfg.llm.num_iters + 1, desc=f'[iteration] {exp_name}', file=sys.stdout, position=tpos):
        iter_ind_root = offspring_root / f'{i_iter:04d}'
        iter_ind_root.mkdir(parents=True, exist_ok=True)

        iter_root = iteration_root / f'{i_iter:04d}'
        iter_root.mkdir(parents=True, exist_ok=True)

        indices = population.sample(iter_root)
        msgs = physicist.get_msgs(population, indices, iter_root / 'messages')
        if i_iter == cfg.llm.num_iters:
            break
        response = physicist.generate(msgs, iter_root / 'choices', iter_root)

        for i_ind, ind_choice in enumerate(tqdm(response.choices, desc=f'[offspring] {exp_name}', file=sys.stdout, position=tpos + 1)):

            try:
                ind_root = iter_ind_root / f'{i_ind:04d}'
                ind_root.mkdir(parents=True, exist_ok=True)

                code_path = ind_choice.dump_root / 'code.py'

                if len(ind_choice.code) == 0:
                    raise RuntimeError('No code generated or generated solution violated format requirements.')

                # train
                train_args = {
                    'tpos': tpos + 2,
                    'path': ind_root,
                    'dataset': cfg.dataset_path,
                    'physics.env.physics.path': code_path
                }
                error = sga.utils.run_exp(base_train_cmds, train_args, unknown_args, my_env)
                if len(error) > 0:
                    raise RuntimeError(error.rsplit('\n', maxsplit=1)[-1])
                losses, params = sga.utils.parse_tensorboard(ind_root)
                states = None
                fitness = min(losses['position'], key=lambda x: x if not math.isnan(x) else float('inf'))
                feedbacks = []

                if cfg.llm.name.startswith('openai-gpt-4'):
                    # evaluate and render
                    for eval_key in ['final']:
                        eval_args = {
                            'is_dataset': False,
                            'tpos': tpos + 2,
                            'path': ind_root / 'eval' / eval_key,
                            'physics.env.physics.path': code_path,
                            'ckpt_path': ind_root / 'ckpt' / f'{eval_key}.pt'
                        }
                        error = sga.utils.run_exp(base_eval_cmds, eval_args, unknown_args, my_env)
                    states = torch.load(ind_root / 'eval' / eval_key / 'state' / 'ckpt.pt', map_location='cpu')
                    feedbacks += get_state_feedback(states, cfg.llm.state_size)
                    feedbacks.append('')  # add a blank line

                feedbacks += get_perf_feedback(losses, params)
                feedback = '\n'.join(feedbacks)
                population.add_offspring(ind_choice, feedback, fitness, losses, params, states, ind_root)
            except Exception as e:
                feedback = str(e)
                fitness = float('inf')
                losses = None
                params = None
                states = None
                population.add_offspring(ind_choice, feedback, fitness, losses, params, states, ind_root)

if __name__ == '__main__':
    main()

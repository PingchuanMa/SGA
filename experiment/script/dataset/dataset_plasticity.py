from pathlib import Path
import subprocess
import os
import sys

from sga.utils import get_root, get_script_parser, dict_to_cmds

root = get_root(__file__)


def main():
    python_path = Path(sys.executable).resolve()
    program_path = root / 'entry' / 'plasticity' / 'eval.py'
    base_cmds = [python_path, program_path]

    base_args = get_script_parser().parse_args()
    base_args = vars(base_args)

    base_args['overwrite'] = True

    my_env = os.environ.copy()
    # my_env['CUDA_VISIBLE_DEVICES'] = str(base_args['gpu'])

    for physics in [
        'sand',
        'water',
        'plasticine',
        'identity',
    ]:
        args = base_args | {
            'path': f'dataset/{physics}',
            'physics.env.physics': physics,
            'is_dataset': True
        }

        cmds = base_cmds + dict_to_cmds(args)
        str_cmds = [str(cmd) for cmd in cmds]
        subprocess.run(str_cmds, shell=False, check=False, env=my_env)

if __name__ == '__main__':
    main()

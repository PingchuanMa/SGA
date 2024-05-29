from pathlib import Path
import subprocess

import numpy as np

import sga
from sga.utils import Candidate


class BaseRenderer(Candidate):
    def __init__(self, cfg: sga.config.physics.render.BaseRenderConfig) -> None:
        super().__init__()

        self.width, self.height = cfg.width, cfg.height
        self.fps = cfg.fps
        self.format = cfg.format

    def make_video(self, image_root: Path, video_path: Path):

        subprocess.run([
            'ffmpeg',
            '-y',
            '-hide_banner',
            '-loglevel', 'error',
            '-framerate', str(self.fps),
            '-i', str(image_root / self.format),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            str(video_path)
        ], check=False)

    @staticmethod
    def cat_videos(input_videos: list[Path], output_video: Path):

        output_video.parent.mkdir(parents=True, exist_ok=True)

        num_videos = len(input_videos)

        if num_videos <= 1:
            raise ValueError('concatenating <=1 videos')

        video_args = []
        for input_video in input_videos:
            video_args.extend(['-i', str(input_video)])

        subprocess.run([
            'ffmpeg',
            '-y',
            '-hide_banner',
            '-loglevel', 'error',
        ] + video_args + [
            '-filter_complex',
            '{}hstack=inputs={}[v]'.format(''.join([f'[{i}:v]' for i in range(num_videos)]), num_videos),
            '-map', '[v]',
            str(output_video)
        ], check=False)

    def __enter__(self) -> 'BaseRenderer':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass

    def save(self, path: Path | str, step: int | str) -> None:
        raise NotImplementedError

    def add_pcd(self, positions: np.ndarray, **kwargs) -> None:
        raise NotImplementedError

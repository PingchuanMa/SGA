from pathlib import Path
import torch
from torch.utils.data import Dataset

class MPMDataset(Dataset):

    def __init__(self, root: str | Path, device: torch.device) -> None:
        super().__init__()

        with torch.no_grad():
            self.ckpt_path = root / 'state' / 'ckpt.pt'
            self.ckpt = torch.load(self.ckpt_path, map_location=device)

            self.xs = self.ckpt['x']
            self.vs = self.ckpt['v']
            self.Cs = self.ckpt['C']
            self.Fs = self.ckpt['F']
            self.stresss = self.ckpt['stress']

    def __len__(self) -> int:
        return self.xs.size(0)

    def __getitem__(self, index):
        return self.xs[index], self.vs[index], self.Cs[index], self.Fs[index], self.stresss[index]

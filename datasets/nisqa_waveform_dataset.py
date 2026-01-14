import os
import torch
import pandas as pd
import torchaudio
from torch.utils.data import Dataset


class NISQAWaveformDataset(Dataset):
    """
    Base Dataset for NISQA.
    Returns:
        waveform : Tensor [1, T]
        mos      : Tensor []
    """

    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        sample_rate: int = 16000,
        duration: float = 5.0,
        debug: bool = False,
    ):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * duration)
        self.debug = debug

        if self.debug:
            print(f"[INFO] CSV loaded: {csv_path}")
            print(f"[INFO] Samples: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def _load_waveform(self, relative_path: str) -> torch.Tensor:
        full_path = os.path.join(self.root_dir, relative_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(full_path)

        waveform, sr = torchaudio.load(full_path)


        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)


        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.sample_rate
            )

        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        else:
            pad = self.max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        return waveform

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        waveform = self._load_waveform(row["filepath_deg"])
        mos = torch.tensor(float(row["mos"]), dtype=torch.float32)

        return waveform, mos

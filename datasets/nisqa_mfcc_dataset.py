import torch
import torchaudio

from datasets.nisqa_waveform_dataset import NISQAWaveformDataset


class NISQAMFCCDataset(NISQAWaveformDataset):
    """
    NISQA Dataset returning MFCC features.

    Returns:
        mfcc : Tensor [1, n_mfcc, T]
        mos  : Tensor []  (normalized to [0,1] if normalize_mos=True)
    """

    def __init__(
        self,
        csv_path,
        root_dir,
        sample_rate=16000,
        duration=5.0,
        n_mfcc=20,
        normalize_mos=True,
        debug=False,
    ):
        super().__init__(
            csv_path=csv_path,
            root_dir=root_dir,
            sample_rate=sample_rate,
            duration=duration,
            debug=debug,
        )

        self.normalize_mos = normalize_mos

        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 64,
            },
        )

    def __getitem__(self, index):
        waveform, mos = super().__getitem__(index)


        mfcc = self.mfcc_transform(waveform)
        mfcc = mfcc[:, :20, :]


        if self.normalize_mos:
            mos = (mos - 1.0) / 4.0

        return mfcc, mos
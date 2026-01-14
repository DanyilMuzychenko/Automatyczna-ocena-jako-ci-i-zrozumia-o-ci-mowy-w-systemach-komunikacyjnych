import torch
import torchaudio

from datasets.nisqa_waveform_dataset import NISQAWaveformDataset


class NISQAMelDataset(NISQAWaveformDataset):
    """
    NISQA Dataset returning log-Mel spectrograms for CNN.

    Returns:
        mel : Tensor [1, n_mels, T]
        mos : Tensor []
    """

    def __init__(
        self,
        csv_path,
        root_dir,
        sample_rate=16000,
        duration=5.0,
        n_mels=64,
        normalize_mos=False,   # ⬅ ВАЖНО
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

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels,
        )

    def __getitem__(self, index):
        waveform, mos = super().__getitem__(index)


        mel = self.mel_transform(waveform)   # [1, n_mels, T]
        mel = torch.log(mel + 1e-9)


        if self.normalize_mos:
            mos = (mos - 1.0) / 4.0   # MOS [1,5] → [0,1]

        return mel, mos

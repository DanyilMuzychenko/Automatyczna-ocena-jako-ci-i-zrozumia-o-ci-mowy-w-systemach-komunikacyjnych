import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model

class NISQAWav2VecDataset(Dataset):
    def __init__(self, csv_path, root_dir, normalize_mos=True, max_len=16000*5):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.normalize_mos = normalize_mos
        self.max_len = max_len

        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base"
        )
        self.model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path):
        wav, sr = torchaudio.load(path)

        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        wav = wav.mean(dim=0)

        if wav.shape[0] > self.max_len:
            wav = wav[:self.max_len]
        else:
            pad = self.max_len - wav.shape[0]
            wav = torch.nn.functional.pad(wav, (0, pad))

        return wav

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.root_dir, row["filepath_deg"])
        mos = torch.tensor(row["mos"], dtype=torch.float32)

        if self.normalize_mos:
            mos = (mos - 1) / 4.0

        wav = self._load_audio(audio_path)

        inputs = self.processor(
            wav,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            features = self.model(**inputs).last_hidden_state.squeeze(0)


        return features, mos

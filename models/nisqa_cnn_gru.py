import torch
import torch.nn as nn

class NISQACNN_GRU(nn.Module):
    def __init__(self, rnn_hidden=128):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )


        self.proj = nn.Linear(64, 512)

        self.gru = nn.GRU(
            input_size=512,
            hidden_size=rnn_hidden,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(rnn_hidden * 2, 1)

    def forward(self, x):
        # x: [B, 1, F, T]
        x = self.cnn(x)           # [B, 64, F', T']
        x = x.mean(dim=2)         # [B, 64, T']
        x = x.permute(0, 2, 1)    # [B, T', 64]

        x = self.proj(x)          # [B, T', 512]

        out, _ = self.gru(x)
        out = out.mean(dim=1)

        return self.fc(out).squeeze(1)

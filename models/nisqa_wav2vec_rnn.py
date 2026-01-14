import torch
import torch.nn as nn

class NISQAWav2VecRNN(nn.Module):
    def __init__(self, rnn_type="gru", hidden=128):
        super().__init__()

        if rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=768,
                hidden_size=hidden,
                batch_first=True,
                bidirectional=True
            )
        else:
            self.rnn = nn.LSTM(
                input_size=768,
                hidden_size=hidden,
                batch_first=True,
                bidirectional=True
            )

        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        # x: [B, T, 768]
        out, _ = self.rnn(x)
        out = out.mean(dim=1)
        return self.fc(out).squeeze(1)

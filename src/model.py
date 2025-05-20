import torch
import torch.nn as nn

class LSTMPolicy(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, hidden_layers=16, output_size=3):
        super(LSTMPolicy, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=hidden_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hx):
        if hx is not None:
            hx = tuple(h.to(x.device) for h in hx)
        x, hx = self.lstm(x, hx)
        x = self.fc(x[:, -1, :])
        return x, hx


import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h, _ = self.lstm(x)
        o = self.linear(h)
        return o


class ExtendedLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.final = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, output_size))

    def forward(self, x):
        h, _ = self.lstm(x)
        o = self.final(h)
        return o



if __name__ == '__main__':
    lstm = LSTM(input_size=171, hidden_size=64, output_size=24)
    x = torch.randn(32, 171)
    o = lstm(x)
    print(o.shape)

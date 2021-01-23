from torch import nn

class Xor(nn.Module):
  def __init__(self):
    super().__init__()
    self.lstm = nn.LSTM(input_size=1, hidden_size=4, num_layers=10)
    self.decoder = nn.Linear(in_features=4, out_features=1)

  def forward(self, x):
    lstm_y, (h_n, c_n) = self.lstm(x)
    y = self.decoder(lstm_y[-1])
    return y

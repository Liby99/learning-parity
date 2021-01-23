import argparse
import pickle
import torch
from torch import optim

from model import Xor

def loss(y, y_predict):
  return (y_predict - y) ** 2

def parser():
  parser = argparse.ArgumentParser(description='Train')
  parser.add_argument("-i", "--input", type=str, default="data/dataset.pkl")
  parser.add_argument("-o", "--output", type=str, default="model/model.pkl")
  parser.add_argument("-l", "--learning-rate", type=float, default=0.001)
  parser.add_argument("-e", "--epoch", type=int, default=1)
  return parser

if __name__ == "__main__":
  args = parser().parse_args()
  dataset = pickle.load(open(args.input, "rb"))
  model = Xor()
  optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
  dataset_size = len(dataset)
  for e in range(args.epoch):
    sum_loss = 0.0
    for ct, (string, y) in enumerate(dataset):
      optimizer.zero_grad()
      x = torch.tensor(string, dtype=torch.float).reshape(-1, 1, 1)
      y_predict = model(x)
      l = loss(y, y_predict)
      sum_loss += l.item()
      print(f"Epoch: {e}, #{ct}/{dataset_size}, avg loss: {sum_loss / (ct + 1)}", end="\r")
      l.backward()
      optimizer.step()
    print(f"Epoch: {e}, avg loss: {sum_loss / (ct + 1)}")
  pickle.dump(model, open(args.output, "wb"))

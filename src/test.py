import pickle
import argparse
import torch

def accuracy(y, y_predict):
  return 1 - abs(y - y_predict)

def parser():
  parser = argparse.ArgumentParser(description='Train')
  parser.add_argument("-m", "--model", type=str, default="model/model.pkl")
  parser.add_argument("-d", "--dataset", type=str, default="data/test.pkl")
  return parser

if __name__ == "__main__":
  args = parser().parse_args()
  model = pickle.load(open(args.model, "rb"))
  dataset = pickle.load(open(args.dataset, "rb"))
  sum_accuracy = 0
  dataset_size = len(dataset)
  for ct, (string, y) in enumerate(dataset):
    x = torch.tensor(string, dtype=torch.float).reshape(-1, 1, 1)
    y_predict = model(x)
    sum_accuracy += accuracy(y, y_predict)
    print(f"#{ct}/{dataset_size}, accuracy: {sum_accuracy.item() / (ct + 1)}", end="\r")
  accuracy = sum_accuracy / len(dataset)
  print(f"Accuracy: {accuracy.item()}")

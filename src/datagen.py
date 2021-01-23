import argparse
import pickle
import random

def generate_one(min_length, max_length):
  length = random.randint(min_length, max_length)
  num_ones = random.randint(0, length)
  string = [0] * num_ones + [1] * (length - num_ones)
  random.shuffle(string)
  parity = num_ones % 2
  return (string, parity)

def generate(amount, min_length, max_length):
  dataset = []
  for i in range(amount):
    dataset.append(generate_one(min_length, max_length))
  return dataset

def parser():
  parser = argparse.ArgumentParser(description='Generate parity dataset')
  parser.add_argument("-o", "--output", type=str, default="data/dataset.pkl")
  parser.add_argument("--amount", type=int, default=1000)
  parser.add_argument("--min-len", type=int, default=10)
  parser.add_argument("--max-len", type=int, default=30)
  return parser

if __name__ == "__main__":
  args = parser().parse_args()
  dataset = generate(args.amount, args.min_len, args.max_len)
  pickle.dump(dataset, open(args.output, "wb"))

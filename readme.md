# Learning Parity with LSTM

This python package tries to learn to compute parity with LSTM. It takes
in a sequence of 0/1s, and produce a single 0/1 representing the parity
bit.

## How to use

1. Generate Training Dataset and Testing Dataset

```
python3 src/datagen.py -o data/train.pkl
python3 src/datagen.py -o data/test.pkl
```

2. Train a model

```
python3 src/train.py -i data/train.pkl -o model/model.pkl
```

3. Test the trained model

```
python3 src/test.py -m model/model.pkl -d data/test.pkl
```

## Expected output

The LSTM model cannot really learn the concept of parity and at the end
it will just randomly guess either 0 or 1, producing a 0.5 accuracy.

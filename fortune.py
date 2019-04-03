"""Main program for training, testing, and interface
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split
import json
import random

import models
import utils

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# === PARAMS ===
HIDDEN_DIM = 64
EMBEDDING_DIM = 64
EPOCHS = 700
BATCH_SIZE = 20

model_path = "checkpoints/checkpoint_600_0.0"
do_train = False
do_eval = True


# load vocab
vocab = models.Vocab()
vocab.load_vocab("words_alpha.txt")

# load fortunes and non-fortunes
max_len = -1
fortunes = []
with open("fortunes.json", "r") as f:
    data = json.loads(f.read())
    for key in data:
        s = utils.clean_sentence(data[key])
        fortunes.append(s)
        if len(s) > max_len:
            max_len = len(s)

nonfortunes = []
with open("nonfortunes.json", "r") as f:
    data = json.loads(f.read())
    for key in data:
        s = utils.clean_sentence(data[key])
        nonfortunes.append(s)
        if len(s) > max_len:
            max_len = len(s)


# convert to tensor
all_data = []
all_targets = []
for s in fortunes:
    all_data.append(vocab.encode(s, max_len+1))
    all_targets.append(torch.tensor([1., 0]))
for s in nonfortunes:
    all_data.append(vocab.encode(s, max_len+1))
    all_targets.append(torch.tensor([0., 1]))

# partition training and test data
train_in, test_in, train_targets, test_targets = train_test_split(all_data, all_targets, test_size=0.33, random_state=42)
train_data = list(zip(train_in, train_targets))
test_data = list(zip(test_in, test_targets))

# === SET UP ===
model = models.FortuneIdentifier(EMBEDDING_DIM, HIDDEN_DIM, vocab.size, max_len+1)
if model_path:
    model.load_state_dict(torch.load(model_path))
    model.eval()

loss_f = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

loss_list = []

# === TRAIN ===
if do_train:
    print("Start training...")
    print(f"Training size: {len(train_data)}")
    for epoch in range(1, EPOCHS+1):
        avg_loss = 0
        for inp, target in random.sample(train_data, BATCH_SIZE):
            model.zero_grad()

            scores = model(inp)

            loss = loss_f(scores, target)
            avg_loss += loss
            loss.backward()
            optimizer.step()

        loss_list.append(avg_loss)

        # save model
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"checkpoints/checkpoint_{epoch}_{round((avg_loss / BATCH_SIZE).item(), 2)}")

        print(f"Epoch {epoch}\tAvg Loss: {avg_loss / BATCH_SIZE}")


# === TEST ===
if do_eval:
    with torch.no_grad():
        if model_path:
            model.load_state_dict(torch.load(model_path))
            model.eval()
        
        print("Start testing...")
        n_correct = 0
        false_neg = 0
        false_pos = 0
        for inp, target in test_data:
            out_t = model(inp)
            if torch.argmax(out_t) == torch.argmax(target):
                n_correct += 1
            elif torch.argmax(out_t) == 0 and torch.argmax(target) != 0:
                false_pos += 1
            elif torch.argmax(out_t) == 1 and torch.argmax(target) != 1:
                false_neg += 1

        print(f"False Neg:\t{false_neg}")
        print(f"False Pos:\t{false_pos}")
        print(f"Correct:\t{n_correct}/{len(test_data)} = {n_correct / len(test_data)}")

# === CMD INTERFACE ===
with torch.no_grad():
    if model_path:
        model.load_state_dict(torch.load(model_path))
        model.eval()

    while True:
        sen = input("> ")
        in_t = vocab.encode(utils.clean_sentence(sen), max_len+1)
        out_t = model(in_t)
        
        print(out_t)
        print(["Fortune", "Not fortune"][torch.argmax(out_t)])

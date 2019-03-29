import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import json
import random
import re

# === MODELS ===
class FortuneIdentifier(nn.Module):
    """Fortune cookie identifier
    """
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_size):
        super(FortuneIdentifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_size = max_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.connected = nn.Sequential(
            nn.Linear(hidden_dim * max_size, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softmax(),
        )

    def forward(self, x):
        embeds = self.embedding(x)
        out, _ = self.lstm(embeds.view(len(x), 1, -1))
        scores = self.connected(out.view(1, -1))
        return scores.squeeze()

# === PRE-PROCCESS ===
class Vocab():
    """Vocab to encode sentences to tensors
    """

    def __init__(self):
        self.idx2word = {
            0:"<NUL>",
            1:"<SOS>",
            2:"<EOS>",
        }
        self.word2idx = {
            "<NUL>":0,
            "<SOS>":1,
            "<EOS>":2,
        }
        self.size = 3

    def load_vocab(self, fname):
        """Load word list into vocab
        """
        with open(fname, "r") as f:
            word_list = f.read().split("\n")
            for i, word in enumerate(word_list):
                if not word in self.word2idx:
                    self.idx2word[i+3] = word
                    self.word2idx[word] = i+3
                    self.size += 1

    def encode(self, sentence, max_size):
        """Encodes list of words to tensor using vocab
        """

        idx_list = [0] * max_size
        i = 0
        for word in sentence:
            try:
                idx_list[i] = self.word2idx[word]
            except KeyError:
                idx_list[i] = random.randint(3, self.size-1)
            i += 1
        idx_list[i] = self.word2idx["<EOS>"]

        return torch.LongTensor(idx_list)


def clean_sentence(data):
    """Cleans sentence for indexing
    """
    data = data.split(" ")
    data = list(map(lambda x: x.lower(), data))
    data = list(map(lambda x: re.sub(r'\W+', '', x), data))
    return data

# load vocab
vocab = Vocab()
vocab.load_vocab("words_alpha.txt")

max_len = -1

# load fortunes
fortunes = []
with open("fortunes.json", "r") as f:
    data = json.loads(f.read())
    for key in data:
        s = clean_sentence(data[key])
        fortunes.append(s)
        if len(s) > max_len:
            max_len = len(s)
            
# load non-fortunes
nonfortunes = []
with open("nonfortunes.json", "r") as f:
    data = json.loads(f.read())
    for key in data:
        s = clean_sentence(data[key])
        nonfortunes.append(s)
        if len(s) > max_len:
            max_len = len(s)


# convert to tensor
all_data = []
for s in fortunes:
    all_data.append((vocab.encode(s, max_len+1), torch.tensor([1., 0])))
for s in nonfortunes:
    all_data.append((vocab.encode(s, max_len+1), torch.tensor([0., 1])))

# partition training and test data
#TODO
train_data = all_data

# === TRAIN ===
HIDDEN_DIM = 64
EMBEDDING_DIM = 64
EPOCHS = 100
BATCH_SIZE = 20

model = FortuneIdentifier(EMBEDDING_DIM, HIDDEN_DIM, vocab.size, max_len+1)
loss_f = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

loss_list = []

print("Start training...")
for epoch in range(EPOCHS):
    avg_loss = 0
    for inp, target in random.sample(train_data, BATCH_SIZE):
        model.zero_grad()

        scores = model(inp)

        loss = loss_f(scores, target)
        avg_loss += loss
        loss.backward()
        optimizer.step()

    loss_list.append(avg_loss)
    print(f"Epoch {epoch}\tAvg Loss: {avg_loss / BATCH_SIZE}")


# === TEST ===
with torch.no_grad():
    pass

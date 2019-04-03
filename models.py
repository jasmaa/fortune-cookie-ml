"""Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random

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

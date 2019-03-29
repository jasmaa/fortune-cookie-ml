import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import json
import re


# === Preprocess ===

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
        for i, word in enumerate(sentence):
            try:
                idx_list[i] = self.word2idx[word]
            except KeyError:
                idx_list[i] = -1 #random.randint(3, self.size-1)
        idx_list[i] = self.word2idx["<EOS>"]

        return torch.FloatTensor(idx_list)
        

# load vocab
vocab = Vocab()
vocab.load_vocab("words_alpha.txt")
    
# load data
processed_fortunes = []
with open("fortunes.json", "r") as f:
    fortunes = json.loads(f.read())
    max_len = -1
    
    for key in fortunes:
        data = fortunes[key].split(" ")
        data = list(map(lambda x: x.lower(), data))
        data = list(map(lambda x: re.sub(r'\W+', '', x), data))

        processed_fortunes.append(data)

        if len(data) > max_len:
            max_len = len(data)

# convert to tensor
sentence_tensors = []
for s in processed_fortunes:
    sentence_tensors.append(vocab.encode(s, max_len+1))

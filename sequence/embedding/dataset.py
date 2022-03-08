import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

class ScopeSequenceDataset(Dataset):
    def __init__(self, fpath, encoder):
        self.encoder = encoder
        with open(fpath, "r") as fin:
            self.lines = fin.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        sline = line.strip().split("\t")
        label_tensor = torch.tensor(np.ones((1,4)))
        label = int(sline[4])
        label_tensor[0,label:] = 0
        return self.encoder(sline[0]), self.encoder(sline[1]), label_tensor

class ScopeDummyDataset(Dataset):
    def __init__(self, encoder):
        self.encoder = encoder
        self.lines = [
            ("ba", "bba","a.1.1.1","a.2.1.1",2),
            ("aba","aab","a.1.1.1","a.1.1.1",4),
        ]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        sline = self.lines[idx]
        label_tensor = torch.ones([1,4], requires_grad=False)
        label = int(sline[4])
        label_tensor[0,label:] = 0
        return self.encoder(sline[0]), self.encoder(sline[1]), label_tensor

def pack_batch(batch):
    lengths = [x.size(0) for x in batch]   # get the length of each sequence in the batch\
    padded = nn.utils.rnn.pad_sequence(batch, batch_first=True)  # padd all sequences

    # pack padded sequece
    packed = nn.utils.rnn.pack_padded_sequence(padded, lengths=lengths, batch_first=True, enforce_sorted=False)

    return packed, lengths

def batchify(batch):
    transposed_data = list(zip(*batch))
    batch1, batch2, labels = transposed_data

    return pack_batch(batch1), pack_batch(batch2), torch.vstack(labels)



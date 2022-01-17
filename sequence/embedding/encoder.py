import itertools
import functools

import numpy as np
import torch

amino_acids = "arndcqeghilkmfpstwyv"
counter = itertools.count()
aa2index = {a: next(counter) for a in amino_acids}
dummy2index = {"a": 0, "b": 1}

def sequence_one_hot_encoder(indexer, sequence):
    dim = len(indexer)
    one_hot_encoded = np.zeros((len(sequence), dim+1))
    for i, aa in enumerate(sequence):
        index = indexer.get(aa, dim)
        one_hot_encoded[i, index] = 1.0
    return torch.tensor(one_hot_encoded, dtype=torch.float)

dummy_encoder = functools.partial(sequence_one_hot_encoder, dummy2index)
aa_encoder = functools.partial(sequence_one_hot_encoder, aa2index)

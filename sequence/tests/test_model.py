import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter

from ..embedding.dataset import ScopeDummyDataset, batchify
from ..embedding.encoder import dummy_encoder
from ..embedding.model import SequenceEmbedder, train_loop, ReluRNN
from ..embedding.loss import structural_similarity_loss

def initialize_parameters(model):
    """
    Just to initialize a simple RNN with fixed values for debugging and understanding PyTorch forward flow
    Important note: will not work with LSTM or GRU, will not work with bias in RNN (or bidirectionality)
    """
    ones = np.ones((model.hidden_lstm_units, model.input_dim))
    for rowi in range(ones.shape[0]):
        for coli in range(ones.shape[1]):
            ones[rowi,coli] = rowi+coli
    model.input_stack[0].weight = Parameter(torch.tensor(
        ones,
        dtype=torch.float
    ))
    model.input_stack[0].bias = Parameter(torch.tensor(0, dtype=torch.float))

    for k in range(model.n_lstm_layers):
        ones = np.ones(getattr(model.rnn, f"weight_hh_l{k}").shape)
        setattr(model.rnn, f"weight_hh_l{k}", Parameter(torch.tensor(ones, dtype=torch.float)))

    ones = np.ones(model.output_stack[0].weight.shape)
    model.output_stack[0].weight = Parameter(torch.tensor(
        ones,
        dtype=torch.float
    ))
    model.output_stack[0].bias = Parameter(torch.tensor(0, dtype=torch.float))

def test_forward_model():
    data = ScopeDummyDataset(dummy_encoder)
    dataloader = DataLoader(data, batch_size=2, collate_fn=batchify)

    model = SequenceEmbedder(4, 3, hidden_lstm_units=10, n_lstm_layers=3, output_dim=10, bidirectional=True, recurrent_layer=ReluRNN)
    model.float()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 200
    for t in range(epochs):
        train_loop(dataloader, model, structural_similarity_loss, optimizer)

    batch_left, batch_right, _ = batchify(data)
    predictions = model.forward((batch_left, batch_right))
    positive_mask = (predictions-0.6)>0
    expected_mask = torch.tensor([[True, True, False, False], [True, True, True, True]], dtype=torch.bool)
    assert(torch.all(expected_mask == positive_mask))
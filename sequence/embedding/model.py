import itertools
import functools
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

ReluRNN = functools.partial(nn.RNN, nonlinearity="relu")

class OrdinalRegression(nn.Module):
    def __init__(self, n_classes):
        super(OrdinalRegression, self).__init__()
        self.n_classes = n_classes
        self.coefficients = Parameter(torch.tensor(np.ones((1, n_classes)), dtype=torch.float), requires_grad=True)
        self.bias = Parameter(torch.tensor(np.zeros((1, n_classes)), dtype=torch.float), requires_grad=True)

    def forward(self, batch):
        if batch.size(1) != 1:
            raise Exception("second dimension of input should be 1")
        expanded_bias = self.bias.expand(batch.size(0),-1)
        return torch.sigmoid(torch.matmul(batch,F.relu(self.coefficients))+expanded_bias)

"""
def uniform_alignment(sequence_embedding1, length1, sequence_embedding2, length2):
    r1 = sequence_embedding1[:length1,:].repeat(length2,1)
    r2 = sequence_embedding2[:length2,:].repeat(length1,1)
    return -(r1-r2).abs().sum()/(length1*length2)
"""

class MeanAlignment(nn.Module):
    def __init__(self):
        super(MeanAlignment, self).__init__()

    def forward(self, embeddings1, embeddings2):
        sum1 = torch.sum(embeddings1, dim=1)
        sum2 = torch.sum(embeddings2, dim=1)
        return -torch.linalg.norm(torch.sub(sum1, sum2), ord=1, dim=1)

class SequenceEmbedder(nn.Module):
    def __init__(self, n_classes, input_dim, hidden_lstm_units=512, n_lstm_layers=1, output_dim=100, bidirectional=True, recurrent_layer=nn.LSTM):
        super(SequenceEmbedder, self).__init__()

        self.n_classes = n_classes
        self.input_dim = input_dim
        self.hidden_lstm_units = hidden_lstm_units
        self.n_lstm_layers = n_lstm_layers
        self.bidirectional = bidirectional
        self.output_dim = output_dim
        self.D = 2 if self.bidirectional else 1

        self.input_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_lstm_units, dtype=torch.float),
            nn.ReLU()
        )

        self.rnn = recurrent_layer(
            input_size=hidden_lstm_units,
            hidden_size=hidden_lstm_units,
            num_layers=n_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            bias=True)

        self._fix_rnn_input_parameters()

        self.output_stack = nn.Sequential(
            nn.Linear(self.D*hidden_lstm_units, output_dim, dtype=torch.float),
            nn.ReLU()
        )

        self.alignment_layer = MeanAlignment()

        self.ordinal_regression = OrdinalRegression(self.n_classes)

    def _fix_rnn_input_parameters(self):
        print("called fix_rnn_input_parameters")

        n_repetitions = 1
        if type(self.rnn) == nn.LSTM:
            n_repetitions = 4
        elif type(self.rnn) == nn.GRU:
            n_repetitions = 3

        eye = np.repeat(np.eye(self.hidden_lstm_units), n_repetitions, axis=0)
        setattr(self.rnn,
                "weight_ih_l0",
                Parameter(torch.tensor(eye, dtype=torch.float), requires_grad=False))

        for i in range(1, self.n_lstm_layers):
            eye = np.repeat(np.eye(self.hidden_lstm_units), n_repetitions, axis=0)
            eye = np.repeat(eye, self.D, axis=1)
            setattr(self.rnn,
                f"weight_ih_l{i}",
                Parameter(torch.tensor(eye, dtype=torch.float), requires_grad=False))


    def embed_sequence(self, batch, lengths):
        #unpack packed sequences in batch by padding: shape L*input_dim -> shape B*T*input_dim
        #where L is the sum of all sequence lenghts in the batch
        #where B is the batch size
        #where T is the length of the longest sequence in the batch
        batch_padded, _ = pad_packed_sequence(batch, batch_first=True)

        #relu transformation of input and packing sequences in batch for recurrent layer
        #shape B*T*input_dim -> L*hidden_lstm_units
        rnn_inputs = self.input_stack(batch_padded)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, lengths=lengths, batch_first=True, enforce_sorted=False)

        #apply recurrent layers and unpack by padding the output of the last recurrent layer
        #lstm_h_0 = torch.tensor(np.zeros((self.D*self.n_lstm_layers, batch.batch_sizes[0], self.hidden_lstm_units)), dtype=torch.float)
        #lstm_c_0 = torch.tensor(np.zeros((self.D*self.n_lstm_layers, batch.batch_sizes[0], self.hidden_lstm_units)), dtype=torch.float)
        rnn_output = self.rnn(rnn_inputs)
        rnn_out_unpacked, _ = pad_packed_sequence(rnn_output[0], batch_first=True)

        #feed output of recurrent layer to Relu unit
        return self.output_stack(rnn_out_unpacked)

    def forward(self, batch):
        (batch1, lengths1), (batch2, lengths2) = batch

        sequence_embeddings1 = self.embed_sequence(batch1, lengths1)
        sequence_embeddings2 = self.embed_sequence(batch2, lengths2)

        stacked_scores = self.alignment_layer(sequence_embeddings1, sequence_embeddings2)
        stacked_scores = stacked_scores.reshape((sequence_embeddings1.size(0),1))

        stacked_predictions = self.ordinal_regression(stacked_scores)

        #TODO: labels should be processed somewhere else
        return stacked_predictions

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batchi, (batch_left, batch_right, labels) in enumerate(dataloader):
        # Compute prediction and loss
        print(f"batch left: {batch_left}")
        predictions = model((batch_left, batch_right))
        loss = loss_fn(predictions, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        optimizer.step()

        if batchi % 100 == 0:
            loss, current = loss.item(), batchi * batch_left[0].batch_sizes[0]
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loss(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    loss = 0

    with torch.no_grad():
        for batch_left, batch_right, labels in dataloader:
            predictions = model((batch_left, batch_right))
            loss += loss_fn(predictions, labels).item()

    loss /= num_batches
    print(f"Avg loss: {loss:>8f} \n")
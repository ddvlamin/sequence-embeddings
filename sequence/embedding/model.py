import itertools
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class OrdinalRegression(nn.Module):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.coefficients = torch.tensor(np.ones((1, n_classes))/n_classes, requires_grad=True, dtype=torch.float)
        self.bias = torch.tensor(np.zeros((1, n_classes)), requires_grad=True, dtype=torch.float)

    def forward(self, batch):
        if batch.size(1) != 1:
            raise Exception("second dimension of input should be 1")
        expanded_bias = self.bias.expand(batch.size(0),-1)
        return torch.sigmoid(torch.matmul(batch,F.relu(self.coefficients))+expanded_bias)

def uniform_alignment(sequence_embedding1, length1, sequence_embedding2, length2):
    r1 = sequence_embedding1[:length1,:].repeat(length2,1)
    r2 = sequence_embedding2[:length2,:].repeat(length1,1)
    return -(r1-r2).abs().sum()/(length1*length2)

class SequenceEmbedder(nn.Module):
    def __init__(self, n_classes, input_dim, hidden_lstm_units=512, n_lstm_layers=1, output_dim=100, bidirectional=True):
        super(SequenceEmbedder, self).__init__()

        self.n_classes = n_classes
        self.input_dim = input_dim
        self.hidden_lstm_units = hidden_lstm_units
        self.n_lstm_layers = n_lstm_layers
        self.bidirectional = bidirectional
        self.output_dim = output_dim

        self.rnn = nn.LSTM(#TODO: adjust between LSTM/GRU/RNN
            input_size=hidden_lstm_units,
            hidden_size=hidden_lstm_units,
            num_layers=n_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            bias=False,
            #nonlinearity="relu" #TODO: adjust between LSTM/GRU/RNN
        )
        self.fix_rnn_input_parameters()

        self.input_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_lstm_units, dtype=torch.float),
            nn.ReLU()
        )

        self.D = 2 if self.bidirectional else 1
        self.output_stack = nn.Sequential(
            nn.Linear(self.D*hidden_lstm_units, output_dim, dtype=torch.float),
            nn.ReLU()
        )

    def _initialize_parameters(self):
        """
        Just to initialize a simple RNN with fixed values for debugging and understanding PyTorch forward flow
        Important note: will not work with LSTM or GRU, will not work with bias in RNN (or bidirectionality)
        """
        ones = np.ones((self.hidden_lstm_units, self.input_dim))
        for rowi in range(ones.shape[0]):
            for coli in range(ones.shape[1]):
                ones[rowi,coli] = rowi+coli
        self.input_stack[0].weight = Parameter(torch.tensor(
            ones,
            dtype=torch.float
        ))
        self.input_stack[0].bias = Parameter(torch.tensor(0, dtype=torch.float))

        for k in range(self.n_lstm_layers):
            ones = np.ones(getattr(self.rnn,f"weight_hh_l{k}").shape)
            setattr(self.rnn,f"weight_hh_l{k}",Parameter(torch.tensor(ones, dtype=torch.float)))

        ones = np.ones(self.output_stack[0].weight.shape)
        self.output_stack[0].weight = Parameter(torch.tensor(
            ones,
            dtype=torch.float
        ))
        self.output_stack[0].bias = Parameter(torch.tensor(0, dtype=torch.float))

    def fix_rnn_input_parameters(self):
        #TODO: adjust between LSTM/GRU/RNN
        eye = np.repeat(np.eye(self.hidden_lstm_units), 4, axis=0)
        setattr(self.rnn,
                "weight_ih_l0",
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
        #print(f"{rnn_inputs}")
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, lengths=lengths, batch_first=True, enforce_sorted=False)
        #print(f"{rnn_inputs}")

        #apply recurrent layers and unpack by padding the output of the last recurrent layer
        #lstm_h_0 = torch.tensor(np.zeros((self.D*self.n_lstm_layers, batch.batch_sizes[0], self.hidden_lstm_units)), dtype=torch.float)
        #lstm_c_0 = torch.tensor(np.zeros((self.D*self.n_lstm_layers, batch.batch_sizes[0], self.hidden_lstm_units)), dtype=torch.float)
        rnn_output = self.rnn(rnn_inputs)
        rnn_out_unpacked, _ = pad_packed_sequence(rnn_output[0], batch_first=True)

        #feed output of recurrent layer to Relu unit
        return self.output_stack(rnn_out_unpacked)

    def forward(self, batch):
        (batch1, lengths1), (batch2, lengths2), labels = batch

        sequence_embeddings1 = self.embed_sequence(batch1, lengths1)
        sequence_embeddings2 = self.embed_sequence(batch2, lengths2)

        batch_size = sequence_embeddings1.size(0)
        chunks1 = sequence_embeddings1.chunk(batch_size, 0)
        chunks2 = sequence_embeddings2.chunk(batch_size, 0)

        alignment_scores = []
        for embedding1, length1, embedding2, length2 in zip(chunks1,
                                                            lengths1,
                                                            chunks2,
                                                            lengths2):
            flattened1 = torch.flatten(embedding1, start_dim=0, end_dim=1)
            flattened2 = torch.flatten(embedding2, start_dim=0, end_dim=1)
            alignment_score = uniform_alignment(flattened1, length1, flattened2, length2)
            alignment_scores.append(alignment_score)

        stacked_scores = torch.tensor(alignment_scores).reshape((len(alignment_scores),1))

        stacked_predictions = OrdinalRegression(self.n_classes).forward(stacked_scores)

        #TODO: labels should be processed somewhere else
        return stacked_predictions, labels

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batchi, batch in enumerate(dataloader):
        # Compute prediction and loss
        out = model(batch)
        loss = loss_fn(out[0], out[1])

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batchi % 100 == 0:
            loss, current = loss.item(), batchi * batch[0][0].batch_sizes[0]
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
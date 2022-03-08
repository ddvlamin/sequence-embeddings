
import torch
from torch.utils.data import DataLoader

from dataset import ScopeDummyDataset, batchify
from encoder import dummy_encoder
from model import SequenceEmbedder, train_loop, ReluRNN
from loss import structural_similarity_loss

if __name__ == "__main__":
    data = ScopeDummyDataset(dummy_encoder)
    dataloader = DataLoader(data, batch_size=2, collate_fn=batchify)
    model = SequenceEmbedder(4, 3, hidden_lstm_units=5, output_dim=5, bidirectional=True, recurrent_layer=ReluRNN)
    model.float()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 200
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(dataloader, model, structural_similarity_loss, optimizer)

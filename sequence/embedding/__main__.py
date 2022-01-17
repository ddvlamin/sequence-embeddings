
import torch
from torch.utils.data import DataLoader

from .dataset import ScopeDummyDataset, batchify
from .encoder import dummy_encoder
from .model import SequenceEmbedder, train_loop
from .loss import structural_similarity_loss

if __name__ == "__main__":
    data = ScopeDummyDataset(dummy_encoder)
    dataloader = DataLoader(data, batch_size=2, collate_fn=batchify)
    model = SequenceEmbedder(4, 3, hidden_lstm_units=2, output_dim=2, bidirectional=True)
    model.float()

    learning_rate = 0.001
    loss_fn = structural_similarity_loss

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(dataloader, model, loss_fn, optimizer)
        #test_loop(test_dataloader, model, loss_fn)
        print("Done!")

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
    model.to("cpu")
    print(model)

    learning_rate = 0.01
    loss_fn = structural_similarity_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for p in model.parameters():
        print(f"{p}")

    epochs = 200
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(dataloader, model, loss_fn, optimizer)
        #print(model.ordinal_regression.coefficients.grad)
        #test_loop(test_dataloader, model, loss_fn)
        print("Done!")
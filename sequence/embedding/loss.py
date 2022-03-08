import torch

def structural_similarity_loss(predictions, labels):
    loss_matrix = torch.mul(torch.log(predictions), labels) + torch.mul(torch.log(1.0-predictions), 1-labels)
    print(f"loss matrix: {loss_matrix}")
    return -torch.mean(torch.sum(loss_matrix, dim=1))

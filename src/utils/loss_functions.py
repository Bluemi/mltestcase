import torch

from utils import describe


def same_loss(diffs, labels: torch.Tensor):
    same_mask = torch.eq(labels.reshape(-1, 1), labels.reshape(1, -1))
    masked_diff = same_mask * diffs
    s_loss = torch.sum(masked_diff) / torch.sum(same_mask)
    return s_loss


def different_loss(diffs, labels, sigma):
    different_mask = torch.ne(labels.reshape(1, -1), labels.reshape(-1, 1))
    masked_diff = torch.exp(-different_mask.to(int) * diffs * sigma)
    d_loss = torch.sum(masked_diff) / torch.sum(different_mask)
    return d_loss


def custom_loss_function(outputs, inputs, embedding, labels, alpha=1.0, beta=1.0, gamma=1.0, sigma=0.5):
    batch_size = outputs.size(0)
    diffs = embedding.reshape(batch_size, 1, -1) - embedding.reshape(1, batch_size, -1)
    diffs = torch.sum(torch.square(diffs), dim=2)
    s_loss = same_loss(diffs, labels)
    d_loss = different_loss(diffs, labels, sigma)
    mse_loss = torch.mean(torch.sum(torch.square(outputs - inputs), dim=1))
    loss = alpha * mse_loss + beta * s_loss + gamma * d_loss
    # print(f'same_loss={s_loss.item()}  diff_loss={d_loss.item()}  mse_loss={mse_loss.item()}')
    return loss

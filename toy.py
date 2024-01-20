import torch

seq_len = 5
k = 3
batch_size = 1
n_heads = 2
matrix = torch.rand((batch_size, n_heads, seq_len, seq_len))
print(matrix)

def get_topk_mask(matrix):
    _, indices = torch.topk(matrix, k, dim=3)

    mask = torch.zeros_like(matrix, dtype=torch.bool)
    mask.scatter_(3, indices, True)
    mask = ~mask
    attention_bias = torch.zeros_like(matrix)
    attention_bias.masked_fill_(mask, -torch.inf)
    return attention_bias


print(get_topk_mask(matrix))

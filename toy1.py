import torch
batch_size = 1
n_heads = 2
s_p = 5

s_r = 3

d = 4

k_r = torch.randint(0, 5, (batch_size, n_heads, s_p, s_r, d))
q_p = torch.randint(0, 5, (batch_size, n_heads, s_p, d))
k_p = torch.randint(0, 5, (batch_size, n_heads, s_p, d))

print(k_r)
print(q_p)
print(k_p)
def prompt_retrieved_attention(q_prompt, k_retrieved):
    q_p_reshaped = q_p.view(batch_size * n_heads * s_p, d, 1)

    result = torch.bmm(k_r.view(batch_size * n_heads * s_p, s_r, d), q_p_reshaped).view(batch_size, n_heads, s_p, s_r)
    return result


def prompt_attention(q_prompt, k_prompt):
    return torch.matmul(q_prompt, k_prompt.transpose(2, 3))

def

print(prompt_retrieved_attention(q_p, k_r).shape)
print(prompt_attention(q_p, k_p).shape)

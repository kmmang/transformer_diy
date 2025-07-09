import torch
from torch import nn

from attention import MultiHeadAttention
from embedding import TransfomerEmbedding


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.d_model = d_model
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        out = self.gamma * (x - mean) / torch.sqrt(var + self.eps)
        return out


if __name__ == "__main__":
    seq = torch.randint(1, 10000, size=(20,)).unsqueeze(0)
    seq_emb = TransfomerEmbedding(10000, 512, 2000, 0.1, 'cpu')(seq)
    out = MultiHeadAttention(512, 8)(seq_emb)
    out = LayerNorm(512)(out)
    print(out)

import torch
from torch import nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from embedding import TransfomerEmbedding
from layernorm import LayerNorm


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden
        self.fc1 = nn.Linear(d_model, self.hidden)
        self.fc2 = nn.Linear(self.hidden, d_model)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, hidden, n_head, drop_prob=0.1):
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden
        self.n_head = n_head
        self.multiattn = MultiHeadAttention(self.d_model, self.n_head)
        self.layernorm1 = LayerNorm(self.d_model)
        self.drop1 = nn.Dropout(drop_prob)
        self.ffn = FeedForward(self.d_model, self.hidden, drop_prob)
        self.layernorm2 = LayerNorm(self.d_model)
        self.drop2 = nn.Dropout(drop_prob)

    def forward(self, x, mask=None):
        _x = x
        x = self.multiattn(x,x,x,mask)
        x = self.drop1(x)
        x = self.layernorm1(x + _x)
        _x = x
        x = self.ffn(x)
        x = self.drop2(x)
        x = self.layernorm2(x + _x)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, vocab_size, max_len, hidden, n_layer, n_head, device, drop_prob=0.1):
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden
        self.n_head = n_head
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding = TransfomerEmbedding(self.vocab_size, self.d_model, self.max_len, drop_prob, device)
        self.layers = nn.ModuleList(
            [EncoderLayer(self.d_model, self.hidden, self.n_head, drop_prob) for _ in torch.arange(n_layer)]
        )

    def forward(self, x, mask=None):
        #batch_size, seq_len, dim = x.shape
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

if __name__ == "__main__":
    seq = torch.randint(1, 10000, size=(64,20))
    print(seq.shape)
    out = Encoder(512, 10000, 2000, 100, 6, 8, 'cpu', 0.1)(seq)
    print(out.shape)

import torch
from torch import nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from embedding import TransfomerEmbedding
from encoder import FeedForward
from layernorm import LayerNorm


class DecoderLayer(nn.Module):
    def __init__(self, d_model, hidden, n_head, drop_prob=0.1):
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden
        self.n_head = n_head
        self.multiattn = MultiHeadAttention(self.d_model, self.n_head)
        self.layernorm1 = LayerNorm(self.d_model)
        self.drop1 = nn.Dropout(drop_prob)
        self.crossattn = MultiHeadAttention(self.d_model, self.n_head)
        self.layernorm2 = LayerNorm(self.d_model)
        self.drop2 = nn.Dropout(drop_prob)
        self.ffn = FeedForward(self.d_model, self.hidden, drop_prob)
        self.layernorm3 = LayerNorm(self.d_model)
        self.drop3 = nn.Dropout(drop_prob)

    def forward(self, en, de, tgt_padding_mask,cross_mask):
        _x = de
        x = self.multiattn(de, de, de, tgt_padding_mask)
        x = self.drop1(x)
        x = self.layernorm1(x + _x)
        _x = x
        x = self.multiattn(x, en, en, cross_mask)
        x = self.drop2(x)
        x = self.layernorm2(x + _x)
        _x = x
        x = self.ffn(x)
        x = self.drop3(x)
        x = self.layernorm3(x + _x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, de_vocab_size, max_len, hidden, n_layer, n_head, device, drop_prob=0.1):
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden
        self.n_head = n_head
        self.de_vocab_size = de_vocab_size
        self.max_len = max_len
        self.embedding = TransfomerEmbedding(self.de_vocab_size, self.d_model, self.max_len, drop_prob, device)
        self.layers = nn.ModuleList(
            [DecoderLayer(self.d_model, self.hidden, self.n_head, drop_prob) for _ in torch.arange(n_layer)]
        )
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(self.d_model, self.de_vocab_size)

    def forward(self, en, de, src_mask, tgt_mask):
        # batch_size, seq_len, dim = x.shape
        de = self.embedding(de)
        for layer in self.layers:
            de = layer(en, de, src_mask, tgt_mask)
        return de

import torch
from torch import nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from decoder import Decoder
from embedding import TransfomerEmbedding
from encoder import FeedForward, Encoder
from layernorm import LayerNorm


class Transformer(nn.Module):
    def __init__(self,
                 src_pad_idx,
                 tgt_pad_idx,
                 d_model,
                 src_vocab_size,
                 tgt_vocab_size,
                 max_len,
                 hidden,
                 n_en_layer,
                 n_de_layer,
                 n_head,
                 device,
                 drop_prob=0.1):
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden
        self.n_head = n_head
        self.max_len = max_len
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.src_vocab_size=src_vocab_size
        self.tgt_vocab_size=tgt_vocab_size
        self.device = device
        self.encoder = Encoder(d_model, self.src_vocab_size, max_len, hidden, n_en_layer, n_head, device, drop_prob)
        self.decoder = Decoder(d_model, self.tgt_vocab_size, max_len, hidden, n_de_layer, n_head, device, drop_prob)
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(self.d_model, self.tgt_vocab_size)

    def get_padding_mask(self, seq):
        padding_mask = seq.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2).float()
        padding_mask = (1.0 - padding_mask) * -1e9  # [PAD]=-inf
        return padding_mask

    def get_causal_mask(self, seq):
        len = seq.shape[-1]
        causal_mask = torch.tril(torch.ones(len, len), diagonal=1).float()
        causal_mask = causal_mask * -1e9  # [PAD]=-inf
        return causal_mask

    def get_cross_mask(self, q,k,q_pad_idx,k_pad_idx):
        q_padding_mask = q.ne(q_pad_idx)
        k_padding_mask = k.ne(k_pad_idx)
        q_padding_mask = q_padding_mask.unsqueeze(-1)
        k_padding_mask = k_padding_mask.unsqueeze(-2)
        cross_mask=(q_padding_mask*k_padding_mask).float()
        cross_mask=(1.0 - cross_mask) * -1e9
        cross_mask=cross_mask.unsqueeze(1)
        return cross_mask

    def get_tgt_mask(self, seq):
        tgt_padding_mask=self.get_padding_mask(seq)
        tgt_causal_mask=self.get_causal_mask(seq)
        tgt_mask = torch.minimum(
            tgt_padding_mask,
            tgt_causal_mask.unsqueeze(0)
        )
        return tgt_mask

    def forward(self, src, tgt):
        src_padding_mask = self.get_padding_mask(src)
        tgt_mask = self.get_tgt_mask(tgt)
        cross_mask =self.get_cross_mask(tgt,src,self.tgt_pad_idx,self.src_pad_idx)
        en = self.encoder(src, src_padding_mask)
        out = self.decoder(en, tgt, tgt_mask, cross_mask)
        out=self.softmax(out)
        out=self.fc(out)
        return torch.argmax(out,dim=-1)


def padding(seq, max_len):
    seq_ = [F.pad(s, (0, max_len - s.shape[1]), value=0) for s in seq]
    seq_ = torch.cat(seq_, 0)
    return seq_


if __name__ == "__main__":
    max_len = 6
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    batch_size = 1
    hidden = 10
    len_src_words = torch.randint(1, max_len, (batch_size,))
    src = [torch.randint(1, src_vocab_size, size=(1,len_src_words[i])) for i in torch.arange(batch_size)]
    src = padding(src,max_len)

    len_tgt_words = torch.randint(1, max_len, (batch_size,))
    tgt = [torch.randint(1, tgt_vocab_size, size=(1,len_tgt_words[i])) for i in torch.arange(batch_size)]
    tgt = padding(tgt, max_len)

    out = Transformer(0, 0, d_model, src_vocab_size, tgt_vocab_size, max_len, hidden, n_en_layer=6, n_de_layer=6,
                    n_head=8, device='cpu')(src, tgt)
    src=torch.Tensor([[1,0,0,0,0,0]])
    print(src)
    print(out)

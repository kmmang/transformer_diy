import torch
from torch import nn
from embedding import TransfomerEmbedding
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, z, mask=None):
        batch_size=x.shape[0]
        x_len, y_len, z_len = x.shape[1],y.shape[1],z.shape[1]
        head_dim = self.d_model // self.n_head
        q = self.w_q(x)
        k = self.w_k(y)
        v = self.w_v(z)
        q = q.view(batch_size, x_len, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch_size, y_len, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch_size, z_len, self.n_head, head_dim).transpose(1, 2)
        score = self.softmax(q @ k.transpose(2, 3) / torch.sqrt(torch.tensor(self.d_model)))
        # print(f"socre的{score.shape}")
        # print(f"mask的{mask.shape}")
        score = score + mask
        out = score @ v
        out = out.transpose(1, 2).contiguous().reshape(batch_size, x_len, -1)
        return out


if __name__ == "__main__":
    max_len = 2000
    vocab_size = 10000
    d_model = 512
    batch_size = 64
    len_words = torch.randint(1, max_len, (batch_size,))
    seq = [torch.randint(1, vocab_size, size=(len_words[i],)).unsqueeze(0) for i in torch.arange(batch_size)]
    seq_ = [F.pad(s, (0, max_len - s.shape[1]), value=0) for s in seq]
    seq_ = torch.cat(seq_, 0)
    padding_mask = (seq_ != 0).unsqueeze(1).unsqueeze(2).float()
    padding_mask = (1.0 - padding_mask) * -1e9  # [PAD]=-inf
    print(padding_mask.shape)
    out = TransfomerEmbedding(vocab_size, d_model, max_len, 0.1, 'cpu')(seq_)
    out = MultiHeadAttention(512, 8)(out, out, out, padding_mask)
    print(out)

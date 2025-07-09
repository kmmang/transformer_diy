import torch
from torch import nn
import torch.nn.functional as F


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super().__init__(vocab_size, d_model, padding_idx=1)

    def forward(self, x):
        # 直接调用父类的 forward 方法（默认行为）
        return super().forward(x)


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len = x.shape
        return self.encoding[:seq_len, :]


class TransfomerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = PositionEmbedding(d_model, max_len, device)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, x):
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(x)
        return self.drop(token_emb + pos_emb)


if __name__ == "__main__":
    max_len = 2000
    vocab_size=10000
    d_model=512
    batch_size=64
    len_words = torch.randint(1, max_len, (batch_size,))
    seq = [torch.randint(1, vocab_size, size=(1,len_words[i],)) for i in torch.arange(batch_size)]
    seq_ = [F.pad(s, (0, max_len - s.shape[1]), value=0) for s in seq]
    seq_ = torch.cat(seq_, 0)
    print(seq_.shape)
    out = TransfomerEmbedding(vocab_size, d_model, max_len, 0.1, 'cpu')(seq_)
    print(out.shape)

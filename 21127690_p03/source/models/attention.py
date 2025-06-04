import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, embed_dim, heads, dropout=0.1):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, inp):
        batch_size, seq_len, embed_dim = inp.size()
        Q = self.query(inp)
        K = self.key(inp)
        V = self.value(inp)

        Q = Q.view(batch_size, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, embed_dim)
        out = self.out(out)
        return out
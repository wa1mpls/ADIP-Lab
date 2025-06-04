import torch.nn as nn
from .attention import Attention

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, mlp_dim, heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = Attention(embed_dim, heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp):
        x = self.norm1(inp)
        x = inp + self.dropout(self.attention(x))
        x = self.norm2(x)
        x = x + self.dropout(self.ff(x))
        return x
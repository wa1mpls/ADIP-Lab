import torch.nn as nn
from .transformer_block import TransformerBlock

class Transformer(nn.Module):
    def __init__(self, embed_dim, mlp_dim, layers, heads, dropout=0.1):
        super(Transformer, self).__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, mlp_dim, heads, dropout)
            for _ in range(layers)
        ])

    def forward(self, inp):
        x = inp
        for block in self.blocks:
            x = block(x)
        return x
import torch
import torch.nn as nn
from .transformer import Transformer
from .classification_head import ClassificationHead

class VisionTransformer(nn.Module):
    def __init__(self, input_size, patch_size, max_len, heads, classes, layers, embed_dim, mlp_dim, channels=3, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        self.patch_to_embedding = nn.Linear(patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.transformer = Transformer(embed_dim, mlp_dim, layers, heads, dropout)
        self.cls_head = ClassificationHead(embed_dim, classes, dropout)

    def forward(self, inp):
        batch_size = inp.size(0)
        x = inp.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, self.num_patches, -1)
        x = self.patch_to_embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding
        x = self.transformer(x)
        cls_output = x[:, 0]
        out = self.cls_head(cls_output)
        return out
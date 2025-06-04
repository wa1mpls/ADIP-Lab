import torch
import unittest
from models.transformer import Transformer

class TestTransformer(unittest.TestCase):
    def test_transformer_output_shape(self):
        batch_size, seq_len, embed_dim = 2, 16, 64
        mlp_dim, heads, layers = 128, 8, 3
        transformer = Transformer(embed_dim, mlp_dim, layers, heads)
        inp = torch.randn(batch_size, seq_len, embed_dim)
        out = transformer(inp)
        self.assertEqual(out.shape, (batch_size, seq_len, embed_dim))

if __name__ == "__main__":
    unittest.main()
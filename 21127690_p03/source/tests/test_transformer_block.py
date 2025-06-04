import torch
import unittest
from models.transformer_block import TransformerBlock

class TestTransformerBlock(unittest.TestCase):
    def test_transformer_block_output_shape(self):
        batch_size, seq_len, embed_dim = 2, 16, 64
        mlp_dim, heads = 128, 8
        block = TransformerBlock(embed_dim, mlp_dim, heads)
        inp = torch.randn(batch_size, seq_len, embed_dim)
        out = block(inp)
        self.assertEqual(out.shape, (batch_size, seq_len, embed_dim))

if __name__ == "__main__":
    unittest.main()
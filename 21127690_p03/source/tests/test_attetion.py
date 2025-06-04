import torch
import unittest
from models.attention import Attention

class TestAttention(unittest.TestCase):
    def test_attention_output_shape(self):
        batch_size, seq_len, embed_dim = 2, 16, 64
        heads = 8
        attn = Attention(embed_dim, heads)
        inp = torch.randn(batch_size, seq_len, embed_dim)
        out = attn(inp)
        self.assertEqual(out.shape, (batch_size, seq_len, embed_dim))

if __name__ == "__main__":
    unittest.main()
import torch
import unittest
from models.classification_head import ClassificationHead

class TestClassificationHead(unittest.TestCase):
    def test_classification_head_output_shape(self):
        batch_size, embed_dim, classes = 2, 64, 10
        head = ClassificationHead(embed_dim, classes)
        inp = torch.randn(batch_size, embed_dim)
        out = head(inp)
        self.assertEqual(out.shape, (batch_size, classes))

if __name__ == "__main__":
    unittest.main()
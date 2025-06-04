import torch
import unittest
from models.vision_transformer import VisionTransformer

class TestVisionTransformer(unittest.TestCase):
    def test_vision_transformer_output_shape(self):
        batch_size, channels, input_size = 2, 3, 32
        params = {
            "input_size": input_size,
            "patch_size": 4,
            "max_len": 100,
            "heads": 8,
            "classes": 10,
            "layers": 6,
            "embed_dim": 64,
            "mlp_dim": 128,
            "channels": channels,
            "dropout": 0.1
        }
        vit = VisionTransformer(**params)
        inp = torch.randn(batch_size, channels, input_size, input_size)
        out = vit(inp)
        self.assertEqual(out.shape, (batch_size, params["classes"]))

if __name__ == "__main__":
    unittest.main()
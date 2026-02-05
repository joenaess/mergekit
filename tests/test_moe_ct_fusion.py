import torch
import unittest
from mergekit.moe.common import fuse_moe_ct_weights

class TestMoECT(unittest.TestCase):
    def test_fusion_math(self):
        base = torch.ones((128, 128)) * 1.0
        expert = torch.ones((128, 128)) * 2.0
        alpha = 0.5
        
        fused = fuse_moe_ct_weights(base, expert, alpha)
        
        # Expected value: 1.0 + 0.5 * (2.0 - 1.0) = 1.5
        expected = torch.ones((128, 128)) * 1.5
        self.assertTrue(torch.allclose(fused, expected))

if __name__ == "__main__":
    unittest.main()
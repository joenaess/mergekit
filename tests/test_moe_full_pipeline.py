import torch
import unittest
from mergekit.moe.common import get_moe_ct_alpha, fuse_moe_ct_weights, fuse_gate_weights

class TestFullPipeline(unittest.TestCase):
    def test_pipeline_math(self):
        # Setup
        layer_idx = 15 # Middle layer
        total_layers = 32
        base_alpha = 1.0
        
        # 1. Test Alpha (U-shape should be low in middle)
        alpha = get_moe_ct_alpha(layer_idx, total_layers, base_alpha, "u_shaped")
        print(f"Pipeline Alpha (L15): {alpha:.4f}")
        self.assertLess(alpha, 0.1)
        
        # 2. Test FFN Fusion
        base_w = torch.ones((10, 10))
        exp_w = torch.zeros((10, 10))
        fused = fuse_moe_ct_weights(base_w, exp_w, alpha)
        # Result should be close to base (1.0) because alpha is near 0
        self.assertGreater(fused.mean(), 0.9)
        
        # 3. Test Gate Bias
        gate_w = torch.ones((8, 128))
        biased_gate = fuse_gate_weights(None, gate_w, 1.1)
        self.assertAlmostEqual(biased_gate.mean().item(), 1.1, places=5)

if __name__ == "__main__":
    unittest.main()
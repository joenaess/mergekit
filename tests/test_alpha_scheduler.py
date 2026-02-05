import unittest
from mergekit.moe.common import get_moe_ct_alpha

class TestAlphaScheduler(unittest.TestCase):
    def test_linear_decrease_curve(self):
        total_layers = 32
        base_alpha = 0.8
        
        # Check first layer (Should be max plasticity)
        alpha_start = get_moe_ct_alpha(0, total_layers, base_alpha, "linear_decrease")
        # Check last layer (Should be max stability/0.0)
        alpha_end = get_moe_ct_alpha(31, total_layers, base_alpha, "linear_decrease")
        
        print(f"\nStrategy: linear_decrease")
        print(f"Layer 0 Alpha: {alpha_start:.4f}")
        print(f"Layer 31 Alpha: {alpha_end:.4f}")
        
        self.assertAlmostEqual(alpha_start, 0.8)
        self.assertAlmostEqual(alpha_end, 0.0)
    
    def test_u_shaped_curve(self):
        total_layers = 32
        base_alpha = 1.0
        
        alpha_start = get_moe_ct_alpha(0, total_layers, base_alpha, "u_shaped")  # Layer 0
        alpha_mid = get_moe_ct_alpha(15, total_layers, base_alpha, "u_shaped")  # Middle
        alpha_end = get_moe_ct_alpha(31, total_layers, base_alpha, "u_shaped")  # Layer 31
        
        print(f"\nStrategy: u_shaped")
        print(f"Layer 0 Alpha: {alpha_start:.4f}")
        print(f"Layer 15 Alpha: {alpha_mid:.4f}")
        print(f"Layer 31 Alpha: {alpha_end:.4f}")

        self.assertAlmostEqual(alpha_start, 1.0)
        self.assertLess(alpha_mid, 0.1) # Should be near 0
        self.assertAlmostEqual(alpha_end, 1.0)

if __name__ == "__main__":
    unittest.main()
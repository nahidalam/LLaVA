import unittest
import torch
from dataclasses import dataclass

import sys
sys.path.insert(0, '/kaggle/working/llava')

from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_encoder.custom_siglip import SiglipVisionTransformerWithRope

@dataclass
class MockModelArgs:
    mm_vision_tower: str
    use_rope_vision: bool = False
    mm_vision_select_layer: int = -1
    mm_vision_select_feature: str = 'patch'


class TestCustomSiglipRope(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the models and dummy data once for all tests."""
        print("Setting up test suite...")
        cls.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        cls.dtype = torch.bfloat16

        print("Loading original SigLIP encoder...")
        original_args = MockModelArgs(mm_vision_tower='google/siglip-base-patch16-256', use_rope_vision=False)
        cls.original_tower = build_vision_tower(original_args)
        cls.original_tower.to(cls.device, dtype=cls.dtype).eval()

        print("Loading custom RoPE-SigLIP encoder...")
        custom_args = MockModelArgs(mm_vision_tower='google/siglip-base-patch16-256', use_rope_vision=True)
        cls.custom_tower = build_vision_tower(custom_args)
        cls.custom_tower.to(cls.device, dtype=cls.dtype).eval()

        cls.dummy_image_unbatched = torch.randn(3, 256, 256, device=cls.device, dtype=cls.dtype)
        
        print(f"Setup complete. Running tests on device: {cls.device}")

    def test_01_model_instantiation(self):
        """Test 1: Ensure the builder returns the correct custom model class."""
        print("\nRunning Test 1: Model Instantiation...")
        self.assertIsInstance(
            self.custom_tower.vision_tower,
            SiglipVisionTransformerWithRope,
            "The vision tower is not an instance of SiglipVisionTransformerWithRope. Check builder.py."
        )
        print("...PASSED")

    def test_02_architecture_modification(self):
        """Test 2: Verify that the learned positional embedding has been removed."""
        print("\nRunning Test 2: Architecture Modification...")
        custom_vision_model = self.custom_tower.vision_tower
        self.assertIsNone(
            custom_vision_model.embeddings.position_embedding,
            "The original position_embedding layer was not removed."
        )
        self.assertTrue(
            hasattr(custom_vision_model, 'rotary_pos_emb'),
            "The new rotary_pos_emb module was not added."
        )
        print("...PASSED")

    def test_03_rope_embedding_shapes(self):
        """Test 3: Check the shape of the generated cos and sin tensors."""
        print("\nRunning Test 3: RoPE Embedding Shapes...")
        custom_vision_model = self.custom_tower.vision_tower
        height, width = self.dummy_image_unbatched.shape[-2:]
        
        with torch.no_grad():
            cos, sin = custom_vision_model._generate_rope_embeddings(height, width, device=self.device)

        patch_size = custom_vision_model.config.patch_size
        num_patches = (height // patch_size) * (width // patch_size)
        head_dim = custom_vision_model.config.hidden_size // custom_vision_model.config.num_attention_heads
        
        expected_shape = (num_patches, head_dim)
        self.assertEqual(cos.shape, expected_shape, f"cos shape is {cos.shape}, expected {expected_shape}")
        self.assertEqual(sin.shape, expected_shape, f"sin shape is {sin.shape}, expected {expected_shape}")
        print("...PASSED")

if __name__ == '__main__':
    unittest.main()
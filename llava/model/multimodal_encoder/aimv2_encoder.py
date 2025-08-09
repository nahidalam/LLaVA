from __future__ import annotations
from typing import List, Union
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Aimv2VisionModel
class Aimv2VisionTower(nn.Module):
    def __init__(
        self,
        vision_tower: str,
        args,
        delay_load: bool = False,
    ) -> None:
        super().__init__()

        self.vision_tower_name: str = vision_tower
        self.select_layer: int = getattr(args, "mm_vision_select_layer", -2)
        self.select_feature: str = getattr(args, "mm_vision_select_feature", "patch")
        self.target_dim: int = getattr(args, "mm_hidden_size", 1024)

        self.delay_load: bool = delay_load
        self.is_loaded: bool = False

        if not delay_load:
            self.load_model()

    @property
    def dtype(self):
        return next(self.parameters()).dtype if self.is_loaded else torch.float32

    @property
    def device(self):
        return next(self.parameters()).device if self.is_loaded else torch.device("cpu")

    @property
    def num_patches(self):
        if hasattr(self, '_num_patches'):
            return self._num_patches
        return 256  # Default 16x16 patches

    @property
    def num_patches_per_side(self):
        if hasattr(self, '_num_patches_per_side'):
            return self._num_patches_per_side
        # Calculate it if not loaded yet
        return 16  # Default for most vision models, will be updated when loaded

    @property
    def config(self):
        if self.is_loaded:
            return self.image_tower.config
        else:
            # Return a minimal config-like object for compatibility
            class MinimalConfig:
                def __init__(self):
                    self.image_size = 224
                    self.patch_size = 14
                    self.hidden_size = 1024
            return MinimalConfig()

    @property
    def hidden_size(self):
        if hasattr(self, '_hidden_size'):
            return self._hidden_size
        return self.target_dim

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    def load_model(self, device_map: Union[str, dict] | None = None, **unused) -> None:
        if self.is_loaded:
            return

        # Use AutoImageProcessor which will load the correct processor for AIMv2
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)

        tower_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.image_tower = Aimv2VisionModel.from_pretrained(
            self.vision_tower_name,
            output_hidden_states=True,
            device_map=device_map,
            torch_dtype=tower_dtype,
        )
        self.image_tower.requires_grad_(False)
        self.is_loaded = True

        cfg = self.image_tower.config
        self._hidden_size: int = cfg.hidden_size

        self.proj = (
            nn.Linear(self._hidden_size, self.target_dim, bias=False)
            if self._hidden_size != self.target_dim
            else nn.Identity()
        )

        # AIMv2 specific configuration
        # AIMv2-large-patch14-224 outputs 256 patches (16x16) without CLS token
        self._num_patches: int = 256  # Fixed for AIMv2-large-patch14-224
        self._num_patches_per_side: int = 16  # Fixed 16x16 grid

    def forward(self, pixel_values: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        if not self.is_loaded:
            self.load_model()

        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values, dim=0)

        tower_device = next(self.image_tower.parameters()).device
        tower_dtype  = next(self.image_tower.parameters()).dtype

        pixel_values = pixel_values.to(device=tower_device, dtype=tower_dtype)

        outs = self.image_tower(pixel_values=pixel_values, output_hidden_states=True)
        hidden = outs.hidden_states[self.select_layer]

        # AIMv2 doesn't have a CLS token, so we don't need to remove anything
        # The output is already just the patch embeddings
        if self.select_feature not in ["patch", "cls_patch"]:
            raise ValueError("select_feature must be 'patch' or 'cls_patch'")

        hidden = self.proj(hidden)
        proj_dtype = getattr(self.proj, "weight", hidden).dtype
        hidden = hidden.to(proj_dtype)
        return hidden
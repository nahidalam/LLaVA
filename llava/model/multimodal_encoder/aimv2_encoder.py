from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Union

# Hugging Face Transformers provides AIM v2 under this name (≥ v4.42).
from transformers import AutoImageProcessor, Aimv2VisionModel


class Aimv2VisionTower(nn.Module):
    """
    Wraps an AIM v2 vision backbone so LLaVA can query hidden states from any
    layer and optionally drop the CLS token. The public API mirrors
    SiglipVisionTower for seamless swap-in.
    """

    def __init__(
        self,
        vision_tower: str,
        args,
        delay_load: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        vision_tower : str
            Hugging Face repo ID or local path, e.g. "apple/aimv2-large-patch14-224".
        args : argparse.Namespace-like
            Must provide:
              • mm_vision_select_layer   – int (which hidden-state layer to take)
              • mm_vision_select_feature – "patch" | "cls_patch"
        delay_load : bool, default False
            If True, defer downloading weights until the first forward pass.
        """
        super().__init__()

        self.vision_tower_name: str = vision_tower
        self.select_layer: int = getattr(args, "mm_vision_select_layer", -2)
        self.select_feature: str = getattr(args, "mm_vision_select_feature", "patch")
        self.delay_load: bool = delay_load

        self.is_loaded: bool = False
        if not delay_load:
            self.load_model()

    # Properties that LLaVA expects on every vision tower
    @property
    def dtype(self):
        return next(self.parameters()).dtype if self.is_loaded else torch.float32

    @property
    def device(self):
        return next(self.parameters()).device if self.is_loaded else torch.device("cpu")

    # Model I/O
    def load_model(self) -> None:
        """Download weights + processor and prepare convenience attributes."""
        if self.is_loaded:
            return

        # Processor handles resize / normalize / to-tensor
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.vision_tower_name
        )

        # Vision backbone
        self.image_tower = Aimv2VisionModel.from_pretrained(
            self.vision_tower_name,
            output_hidden_states=True,
        )
        self.image_tower.requires_grad_(False)  # freeze backbone
        self.is_loaded = True

        # Metadata for downstream heads
        cfg = self.image_tower.config
        self.hidden_size: int = cfg.hidden_size
        patch = getattr(cfg, "patch_size", 14)
        img_sz = getattr(cfg, "image_size", 224)
        self.num_patches: int = (img_sz // patch) ** 2  # e.g. 14×14 = 196

        # Zero-vector for images that are masked / absent
        self.register_buffer(
            "dummy_feature",
            torch.zeros(1, self.hidden_size, dtype=torch.float32),
            persistent=False,
        )

    # Forward
    def forward(
        self,
        pixel_values: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pixel_values
            Either a single 4-D tensor (B, 3, H, W) or a list of tensors
            of shape (1, 3, H, W). LLaVA sometimes passes a list.

        Returns
        -------
        torch.Tensor
            (B, N, hidden_size) where N = num_patches  or  num_patches+1
            depending on select_feature.
        """
        if not self.is_loaded:
            self.load_model()

        # Convert list → batch.
        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values, dim=0)

        # Forward through backbone.
        outs = self.image_tower(
            pixel_values=pixel_values,
            output_hidden_states=True,
        )
        hidden = outs.hidden_states[self.select_layer]  # (B, 1+N, D)

        # Drop CLS if requested
        if self.select_feature == "patch":
            hidden = hidden[:, 1:]           # keep only patch tokens
        elif self.select_feature == "cls_patch":
            pass                             # keep CLS + patches
        else:
            raise ValueError(
                f"select_feature must be 'patch' or 'cls_patch', got {self.select_feature}"
            )

        return hidden
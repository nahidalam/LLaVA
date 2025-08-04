from __future__ import annotations
import torch
import torch.nn as nn
from typing import List, Union
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
        self.delay_load: bool = delay_load

        # Target embedding dim expected by the text projector (usually 1024)
        self.target_dim: int = getattr(args, "mm_hidden_size", 1024)

        self.is_loaded: bool = False
        if not delay_load:
            self.load_model()

    @property
    def dtype(self):
        return next(self.parameters()).dtype if self.is_loaded else torch.float32

    @property
    def device(self):
        return next(self.parameters()).device if self.is_loaded else torch.device("cpu")

    def load_model(self, device_map=None, **unused) -> None:
        if self.is_loaded:
            return

        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)
        self.image_tower = Aimv2VisionModel.from_pretrained(
            self.vision_tower_name, output_hidden_states=True
        )
        torch_dtype=torch.float32,
        self.image_tower.requires_grad_(False)
        self.is_loaded = True

        cfg = self.image_tower.config
        self.hidden_size: int = cfg.hidden_size  # 768 for aim‑v2‑base

        # 768 ➜ 1024 adapter if needed
        if self.hidden_size != self.target_dim:
            self.proj = nn.Linear(self.hidden_size, self.target_dim, bias=False)
        else:
            self.proj = nn.Identity()

        patch = getattr(cfg, "patch_size", 14)
        img_sz = getattr(cfg, "image_size", 224)
        self.num_patches: int = (img_sz // patch) ** 2

        self.register_buffer(
            "dummy_feature",
            torch.zeros(1, self.target_dim, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, pixel_values: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        if not self.is_loaded:
            self.load_model()

        if pixel_values.dtype != torch.float32:
            pixel_values = pixel_values.float()

        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values, dim=0)

        outs = self.image_tower(pixel_values=pixel_values, output_hidden_states=True)
        hidden = outs.hidden_states[self.select_layer]  # (B, 1+N, 768)

        if self.select_feature == "patch":
            hidden = hidden[:, 1:]  # drop CLS token
        elif self.select_feature != "cls_patch":
            raise ValueError("select_feature must be 'patch' or 'cls_patch'")

        hidden = self.proj(hidden)  # (B, N, 1024)
        return hidden
# coding=utf-8
# Copyright 2021 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch CLIP-Vision Model with 2D ROPE - Reference from Qwen2.5VL-M-ROPE """

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import ModelOutput, auto_docstring, can_return_tuple, logging, torch_int
from transformers import CLIPConfig, CLIPTextConfig
from llava.model.multimodal_encoder.clip_custom_config import CLIPVisionCustomConfig

logger = logging.get_logger(__name__)

class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionCustomConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.use_vision_rope_2d = getattr(config, "use_vision_rope_2d", False)

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        if not self.use_vision_rope_2d:
            self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
            self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False) ## position_ids - 0 to 16
            print("use_ope is False")
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 1
        position_embedding = self.position_embedding.weight.unsqueeze(0)
        num_positions = position_embedding.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        class_pos_embed = position_embedding[:, :1]
        patch_pos_embed = position_embedding[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)
		
	#Reference from src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py - Qwen2_5_VisionTransformerPretrainedModel.rot_pos_emb(grid_thw)
    def build_2d_positions(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        grid_h = height // self.patch_size  # height=336, patch_size = 14, grid_h = 24
        grid_w = width // self.patch_size   # width=336, patch_size = 14, grid_w = 24
        rows = torch.arange(grid_h, device=device) # rows = [0,1,2.......23]
        cols = torch.arange(grid_w, device=device)  # cols = [0,1,2.......23]
        rows = rows.repeat_interleave(grid_w)          # rows = [0*24 times,1*24 times.....23*24times]
        cols = cols.repeat(grid_h)                     # cols = [0 to 23 * 24 times]
        # prepend (0,0) for CLS so CLS gets zero-angle rotation - CLIP prepends CLS token
        pos = torch.stack([rows, cols], dim=-1)        # (576,2) ## x and y positions
        pos = torch.cat([torch.zeros(1, 2, device=device, dtype=pos.dtype), pos], dim=0)  # (1+576,2)
        return pos.to(dtype=dtype)

    def forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=False) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape  ## torch.Size([1, 3, 336, 336])
        if not interpolate_pos_encoding and (height != self.image_size or width != self.image_size):
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size}*{self.image_size})."
            )
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, gridw, gridh] ## torch.Size([1, 1024, 24, 24])
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2) ## torch.Size([1, 576, 1024])
        class_embeds = self.class_embedding.expand(batch_size, 1, -1) ## torch.Size([1, 1, 1024])
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1) ## torch.Size([1, 577, 1024])
        ## Original positional embeddings from CLIP
        if not self.use_vision_rope_2d:
            if interpolate_pos_encoding:
                embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
            else:
                embeddings = embeddings + self.position_embedding(self.position_ids)
            return embeddings
        else:
            ## 2D Rope position input_ids
            rope_pos = self.build_2d_positions(height, width, embeddings.device, embeddings.dtype)
            rope_pos = rope_pos.unsqueeze(0).expand(batch_size, -1, -1) ## torch.Size([1, 577, 2])
            return embeddings, rope_pos



def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    output_attentions: bool = True,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights

#Reference from src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py - apply_rotary_pos_emb_vision(q,k,cos,sin) and class Qwen2_5_VisionRotaryEmbedding
def apply_rope_2d(q: torch.Tensor, k: torch.Tensor, rope_pos: torch.Tensor, theta: float):
    # q,k: (B,H,T,Dh), rope_pos: (B,T,2) with (row, col). Require Dh % 4 == 0
    B, H, T, Dh = q.shape
    if Dh % 4 != 0:
        raise ValueError(f"RoPE-2D requires head_dim divisible by 4, got {Dh}")
    half = Dh // 2
    quarter = Dh // 4
    dtype = q.dtype
    device = q.device

	## Reference from src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py - rotate_half()
    def rotate_half(x):
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    def build_inv_freq(n: int):
        idx = torch.arange(n, device=device, dtype=dtype)
        return (theta ** (-2 * idx / n)).to(dtype)

    # split dims for height and width
    q_h, q_w = q[..., :half], q[..., half:]
    k_h, k_w = k[..., :half], k[..., half:]

    inv_h = build_inv_freq(quarter)  # (quarter,)
    inv_w = build_inv_freq(quarter)

    # positions
    pos_h = rope_pos[..., 0].to(dtype=dtype).unsqueeze(-1)  # (B,T,1)
    pos_w = rope_pos[..., 1].to(dtype=dtype).unsqueeze(-1)

    # angles
    ang_h = pos_h * inv_h                                   # (B,T,quarter)
    ang_w = pos_w * inv_w

    cos_h = torch.repeat_interleave(ang_h.cos(), 2, dim=-1).unsqueeze(1)  # (B,1,T,half)
    sin_h = torch.repeat_interleave(ang_h.sin(), 2, dim=-1).unsqueeze(1)
    cos_w = torch.repeat_interleave(ang_w.cos(), 2, dim=-1).unsqueeze(1)  # (B,1,T,half)
    sin_w = torch.repeat_interleave(ang_w.sin(), 2, dim=-1).unsqueeze(1)

    q_h = q_h * cos_h + rotate_half(q_h) * sin_h
    k_h = k_h * cos_h + rotate_half(k_h) * sin_h
    q_w = q_w * cos_w + rotate_half(q_w) * sin_w
    k_w = k_w * cos_w + rotate_half(k_w) * sin_w

    q = torch.cat([q_h, q_w], dim=-1)
    k = torch.cat([k_h, k_w], dim=-1)
    return q, k

class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Union[CLIPVisionCustomConfig, CLIPTextConfig]):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_vision_rope_2d = config.use_vision_rope_2d
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)


    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_pos: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_vision_rope_2d: Optional[bool] = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, -1, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, -1, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, -1, self.head_dim).transpose(1, 2)

        ## Rotate Q/K per 2D grid
        if self.use_vision_rope_2d and rope_pos is not None:
            queries, keys = apply_rope_2d(queries, keys, rope_pos, getattr(self.config, "rope_theta", 10000.0))

        if self.config._attn_implementation == "flash_attention_2":
            self.is_causal = causal_attention_mask is not None
        else:
            if attention_mask is not None and causal_attention_mask is not None:
                attention_mask = attention_mask + causal_attention_mask
            elif causal_attention_mask is not None:
                attention_mask = causal_attention_mask

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
            output_attentions=output_attentions,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Union[CLIPVisionCustomConfig, CLIPTextConfig]):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        rope_pos: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            rope_pos = rope_pos,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


@auto_docstring
class CLIPPreTrainedModel(PreTrainedModel):
    config_class = CLIPConfig
    base_model_prefix = "clip"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn_2 = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, CLIPVisionEmbeddings):
            factor = self.config.initializer_factor
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            if not self.config.use_vision_rope_2d and getattr(module, "position_embedding", None) is not None:  ## Need to check this!!
                print("CLIPPreTrainedModel")
                nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        elif isinstance(module, CLIPAttention):
            factor = self.config.initializer_factor
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, CLIPMLP):
            factor = self.config.initializer_factor
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)

        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class CLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    @can_return_tuple
    def forward(
        self,
        inputs_embeds,
        rope_pos: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutput:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                causal_attention_mask,
                rope_pos=rope_pos,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionCustomConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.use_vision_rope_2d = config.use_vision_rope_2d

        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        use_vision_rope_2d: Optional[bool] = False
    ) -> BaseModelOutputWithPooling:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        if self.use_vision_rope_2d:
            hidden_states, rope_pos = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        else:
            hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
            rope_pos = None

        hidden_states = self.pre_layrnorm(hidden_states) ## torch.Size([1, 577, 1024] (batch_size, sequence_length, hidden_size)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            rope_pos = rope_pos,    ## Add rotary positional ids
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The vision model from CLIP without any head or projection on top.
    """
)
class CLIPVisionModel(CLIPPreTrainedModel):
    config_class = CLIPVisionCustomConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPVisionCustomConfig):
        super().__init__(config)
        self.vision_model = CLIPVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> BaseModelOutputWithPooling:
        r"""
        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPVisionModel

        >>> model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

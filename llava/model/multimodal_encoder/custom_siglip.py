from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

# Import original SigLIP classes to inherit from them
from transformers.models.siglip.modeling_siglip import (
    SiglipAttention,
    SiglipEncoder,
    SiglipEncoderLayer,
    SiglipVisionTransformer,
    SiglipVisionConfig,
    BaseModelOutputWithPooling,
    BaseModelOutput,
)

# Import our new helper functions
from .rope_vision import VisionRotaryEmbedding, apply_rotary_pos_emb_vision

class SiglipAttentionWithRope(SiglipAttention):
    """Modified SiglipAttention to accept and apply 2D RoPE."""
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # New argument
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        print(f"[DEBUG SiglipAttentionWithRope] hidden_states.shape: {hidden_states.shape}")
        
        batch_size, seq_length, embed_dim = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            print(f"[DEBUG SiglipAttentionWithRope] Applying RoPE. Q shape: {query_states.shape}, Cos shape: {cos.shape}")
            query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_length, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, None


class SiglipEncoderWithRope(SiglipEncoder):
    """
    Modified SiglipEncoder to accept and pass down position_embeddings.
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__(config)
        # Replace the original layers with our RoPE-compatible layers
        self.layers = nn.ModuleList([SiglipEncoderLayerWithRope(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # Accept the new argument
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutput:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for i, encoder_layer in enumerate(self.layers):
            print(f"[DEBUG SiglipEncoderWithRope] Forwarding to layer {i}")
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            # Pass position_embeddings down to each layer
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                position_embeddings=position_embeddings,
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

class SiglipEncoderLayerWithRope(SiglipEncoderLayer):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__(config)
        self.self_attn = SiglipAttentionWithRope(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
    ) -> tuple[torch.FloatTensor]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
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

class SiglipVisionTransformerWithRope(SiglipVisionTransformer):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__(config)
        
        # 1. Remove the original learned positional embeddings
        self.embeddings.position_embedding = None
        self.embeddings.position_ids = None

        # 2. Instantiate our new RoPE module
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        
        # 3. Replace the encoder layers with our modified version
        # self.encoder.layers = nn.ModuleList([SiglipEncoderLayerWithRope(config) for _ in range(config.num_hidden_layers)])
        self.encoder = SiglipEncoderWithRope(config)
    
    @property
    def device(self):
        # A robust way to get the device of a module is to check one of its parameters.
        return self.embeddings.patch_embedding.weight.device
    
    @property
    def dtype(self):
        # A robust way to get the dtype of a module is to check one of its parameters.
        return self.embeddings.patch_embedding.weight.dtype

    def _generate_rope_embeddings(self, height, width, device):
        patch_size = self.config.patch_size
        grid_h, grid_w = height // patch_size, width // patch_size

        hpos_ids = torch.arange(grid_h, device=device).unsqueeze(1).expand(-1, grid_w).flatten()
        wpos_ids = torch.arange(grid_w, device=device).unsqueeze(0).expand(grid_h, -1).flatten()
        
        pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1)
        max_grid_size = max(grid_h, grid_w)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        
        h_emb = rotary_pos_emb_full[pos_ids[:, 0]]
        w_emb = rotary_pos_emb_full[pos_ids[:, 1]]

        # The original `emb` was shape (num_patches, 64).
        # The `apply_rotary_pos_emb_vision` function expects an embedding of half the head dimension,
        # as it will be duplicated to create the full dimension inside the rotation.
        # We now correctly combine the H and W embeddings for a 2D rotation.
        # The final embedding should be duplicated for the full head dimension.
        emb = torch.cat((h_emb, w_emb), dim=-1)
        
        # The RoPE dimension is half the head dimension. We need to duplicate it.
        emb = torch.cat((emb, emb), dim=-1)
        return emb.cos(), emb.sin()

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False, # This will be ignored
    ):
        print(f"[DEBUG SiglipVisionTransformerWithRope] Input pixel_values.shape: {pixel_values.shape}")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        # 1. Get patch embeddings WITHOUT positional embeddings
        hidden_states = self.embeddings.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        
        # # 2. Generate RoPE embeddings on the fly
        # height, width = pixel_values.shape[-2:]
        # position_embeddings = self._generate_rope_embeddings(height, width, device=pixel_values.device)
        
        # # 3. Manually create attention mask if not using Flash Attention
        # # This is a simple mask that allows all tokens to attend to all other tokens.
        # batch_size, seq_len, _ = hidden_states.shape
        # attention_mask = torch.ones((batch_size, seq_len), device=pixel_values.device)

        # The RoPE embeddings are generated for a single image grid.
        # We then expand them to match the batch size.
        height, width = pixel_values.shape[-2:]
        cos, sin = self._generate_rope_embeddings(height, width, device=pixel_values.device)
        
        # Expand for the batch dimension. Shape changes from (seq_len, dim) to (batch_size, seq_len, dim)
        batch_size = pixel_values.shape[0]
        position_embeddings = (
            cos.unsqueeze(0).expand(batch_size, -1, -1),
            sin.unsqueeze(0).expand(batch_size, -1, -1)
        )
        print(f"[DEBUG SiglipVisionTransformerWithRope] Generated position_embeddings shapes: cos={position_embeddings[0].shape}, sin={position_embeddings[1].shape}")

        attention_mask = None
        # attention_mask = torch.ones((batch_size, seq_len), device=pixel_values.device)
        
        # 4. Forward through the encoder, passing the RoPE embeddings
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)
        
        pooled_output = self.head(last_hidden_state) if self.use_head else None

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
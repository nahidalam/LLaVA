import torch
import torch.nn as nn
import logging
from functools import reduce
import operator
from transformers.models.clip.modeling_clip import CLIPAttention

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_2d_rope(q, k, cos_sin_cache):
    """Applies 2D Rotary Position Embedding to Query and Key tensors."""
    cos, sin = cos_sin_cache
    # cos/sin cache has shape (seq_len, head_dim)
    # q/k have shape (batch_size, num_heads, seq_len, head_dim)
    # We need to unsqueeze cache to match: (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RoPE2DEmbedding(nn.Module):
    """
    Generates the cos/sin cache for 2D RoPE using the "Concatenated Embeddings" method.
    """
    def __init__(self, head_dim, grid_dims=(16, 16), base=10000):
        super().__init__()
        
        if head_dim % 4 != 0:
            raise ValueError("head_dim must be divisible by 4 for 2D RoPE.")

        self.grid_h, self.grid_w = grid_dims
        self.head_dim = head_dim
        
        pos_dim_per_axis = head_dim // 2
        
        inv_freq = 1.0 / (base ** (torch.arange(0, pos_dim_per_axis, 2).float() / pos_dim_per_axis))
        
        max_grid_dim = max(self.grid_h, self.grid_w)
        positions = torch.arange(max_grid_dim, dtype=torch.float32)
        
        freqs_1d = torch.einsum("i,j->ij", positions, inv_freq)
        
        emb_1d_lookup = torch.cat((freqs_1d, freqs_1d), dim=-1)
        

        h_indices = torch.arange(self.grid_h).unsqueeze(1).expand(-1, self.grid_w)
        w_indices = torch.arange(self.grid_w).unsqueeze(0).expand(self.grid_h, -1)

        h_embeds = emb_1d_lookup[h_indices]
        w_embeds = emb_1d_lookup[w_indices]
        
        emb_2d = torch.cat((h_embeds, w_embeds), dim=-1)
        
        emb_flat = emb_2d.reshape(-1, self.head_dim)
        
        self.register_buffer("cos_cache", emb_flat.cos(), persistent=False)
        self.register_buffer("sin_cache", emb_flat.sin(), persistent=False)

    def forward(self):
        return self.cos_cache, self.sin_cache

class RoPEVisionAttention(nn.Module):
    """A replacement attention module that uses 2D RoPE."""
    def __init__(self, original_attention_module, grid_dims=(24, 24)):
        super().__init__()
        
        # Copy essential attributes and layers from the original module
        self.config = getattr(original_attention_module, 'config', None)
        self.embed_dim = original_attention_module.embed_dim
        self.num_heads = original_attention_module.num_heads
        self.head_dim = original_attention_module.head_dim
        self.scale = self.head_dim**-0.5
        
        self.k_proj = original_attention_module.k_proj
        self.v_proj = original_attention_module.v_proj
        self.q_proj = original_attention_module.q_proj
        self.out_proj = original_attention_module.out_proj
        
        # Instantiate 2D RoPE module
        self.rope_2d = RoPE2DEmbedding(self.head_dim, grid_dims)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, attention_mask=None, causal_attention_mask=None, output_attentions=False):
        bsz, seq_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = self._shape(query_states, seq_len, bsz)
        key_states = self._shape(key_states, seq_len, bsz)
        value_states = self._shape(value_states, seq_len, bsz)

        cos_sin_cache = self.rope_2d()

        # The cache length corresponds to the number of patches (e.g., 576 for a 24x24 grid)
        num_patches = cos_sin_cache[0].shape[0]

        # Check if a [CLS] token is present by comparing sequence lengths
        if seq_len > num_patches:
            # This is a CLIP-style model with a [CLS] token
            if seq_len != num_patches + 1:
                raise ValueError(
                    f"Input sequence length ({seq_len}) is not equal to number of patches ({num_patches}) + 1 ([CLS] token). "
                    "Please check your architecture."
                )
            
            # Separate the [CLS] token from the patch tokens
            query_cls = query_states[:, :, :1, :]
            key_cls = key_states[:, :, :1, :]
            
            query_patches = query_states[:, :, 1:, :]
            key_patches = key_states[:, :, 1:, :]

            query_patches, key_patches = apply_2d_rope(query_patches, key_patches, cos_sin_cache)
            
            # Concatenate the [CLS] token back with the rotated patch tokens
            query_states = torch.cat((query_cls, query_patches), dim=2)
            key_states = torch.cat((key_cls, key_patches), dim=2)
        elif seq_len == num_patches:
            query_states, key_states = apply_2d_rope(query_states, key_states, cos_sin_cache)
        
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states, key_states, value_states = [t.view(*proj_shape) for t in [query_states, key_states, value_states]]

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2)) * self.scale
        
        if attention_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, seq_len, seq_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, seq_len, seq_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        
        attn_output = torch.bmm(attn_weights, value_states)
        attn_output = attn_output.view(bsz, self.num_heads, seq_len, self.head_dim).transpose(1, 2).reshape(bsz, seq_len, embed_dim)
        
        attn_output = self.out_proj(attn_output)
        
        attn_weights_output = attn_weights if output_attentions else None

        return attn_output, attn_weights_output

def get_module_by_name(module, name):
    """
    Access a nested module using a dot-separated name string.
    Example: get_module_by_name(model, 'vision_model.embeddings')
    """
    return reduce(getattr, name.split('.'), module)

def convert_vision_encoder_to_rope(
    encoder: nn.Module,
    attention_module_class: type,
    positional_embedding_path: str,
    grid_dims: tuple = (24, 24)
) -> nn.Module:
    """
    Recursively traverses a vision encoder and replaces standard attention
    modules with RoPE-enabled attention modules in a generalizable way.

    Args:
        encoder (nn.Module): 
            The vision encoder model (e.g., CLIPVisionModel).
        attention_module_class (type): 
            The class of the attention modules to be replaced (e.g., CLIPAttention).
        positional_embedding_path (str): 
            A dot-separated string path to the positional embedding layer 
            (e.g., 'vision_model.embeddings.position_embedding').
        grid_dims (tuple): 
            The (height, width) of the patch grid.
            
    Returns:
        nn.Module: The modified encoder with RoPE attention.
    """
    logging.info(f"Attempting to convert {encoder.__class__.__name__} to use 2D RoPE.")
    logging.info(f"Targeting attention module: {attention_module_class.__name__}")
    logging.info(f"Targeting positional embedding path: {positional_embedding_path}")

    # Disable original positional embeddings using the provided path
    try:
        pos_embed_module = get_module_by_name(encoder, positional_embedding_path)
        
        if hasattr(pos_embed_module, 'weight'):
            pos_embed_module.weight.data.zero_()
            pos_embed_module.weight.requires_grad = False
            logging.info(f"Successfully disabled positional embeddings at '{positional_embedding_path}'.")
        else:
            logging.warning(f"Module at '{positional_embedding_path}' found, but it has no '.weight' attribute to disable.")

    except AttributeError:
        logging.error(f"Could not find the positional embedding module at the specified path: '{positional_embedding_path}'. Conversion might fail or be incorrect.")
        raise AttributeError(f"Path '{positional_embedding_path}' not found in encoder.")

    # Recursively find and replace attention modules
    def _recursive_replace(module: nn.Module):
        for name, child_module in module.named_children():
            if isinstance(child_module, attention_module_class):
                # If the child is an instance of the target class, replace it.
                logging.info(f"Replacing attention module at: {name} ({child_module.__class__.__name__})")
                new_attention_module = RoPEVisionAttention(child_module, grid_dims)
                setattr(module, name, new_attention_module)
            elif len(list(child_module.children())) > 0:
                # If it's not the target but has children, recurse into it.
                _recursive_replace(child_module)

    _recursive_replace(encoder)
    
    logging.info("Conversion complete.")
    return encoder
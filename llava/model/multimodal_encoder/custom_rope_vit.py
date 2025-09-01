import torch
import torch.nn as nn
import logging
from functools import reduce
import operator
from transformers.models.clip.modeling_clip import CLIPAttention

# Helper function for RoPE
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
    Generates the cos/sin cache for 2D RoPE.
    The first half of the head dimension is for height, the second half is for width.
    """
    def __init__(self, head_dim, grid_dims=(24, 24), base=10000):
        super().__init__()
        self.grid_h, self.grid_w = grid_dims
        self.head_dim = head_dim
        
        # Ensure head_dim is even and divisible by 2 for H and W dimensions
        if head_dim % 4 != 0:
            raise ValueError("head_dim must be divisible by 4 for 2D RoPE.")
        
        # 1. Define the dimensionality for each spatial axis (Height and Width)
        # Each axis gets half of the total head dimension.
        pos_dim_per_axis = head_dim // 2
        
        # The number of frequency bands is half of the dimension for each axis.
        freq_dim = pos_dim_per_axis // 2
        
        # 2. Create a 1D RoPE lookup table
        # This table needs to be large enough for the largest grid dimension.
        max_grid_dim = max(self.grid_h, self.grid_w)
        
        # Standard RoPE frequency calculation
        inv_freq = 1.0 / (base ** (torch.arange(0, freq_dim, 2).float() / freq_dim))
        positions = torch.arange(max_grid_dim, dtype=torch.float32)
        
        # Create the 1D frequency table
        freqs_1d = torch.einsum("i,j->ij", positions, inv_freq) # Shape: (max_grid_dim, freq_dim/2)
        
        # This is our 1D lookup table for one spatial dimension (e.g., height)
        # Shape: (max_grid_dim, pos_dim_per_axis)
        emb_1d_lookup = torch.cat((freqs_1d, freqs_1d), dim=-1)

        # 3. Create coordinate indices for the entire grid
        h_indices = torch.arange(self.grid_h).unsqueeze(1).expand(-1, self.grid_w) # Shape: (grid_h, grid_w)
        w_indices = torch.arange(self.grid_w).unsqueeze(0).expand(self.grid_h, -1) # Shape: (grid_h, grid_w)

        # 4. "Look up" the H and W embeddings from the 1D table
        h_embeds = emb_1d_lookup[h_indices] # Shape: (grid_h, grid_w, pos_dim_per_axis)
        w_embeds = emb_1d_lookup[w_indices] # Shape: (grid_h, grid_w, pos_dim_per_axis)
        
        # 5. Concatenate to form the final 2D embedding and flatten
        # This is the core of the "Concatenated Embeddings" method.
        emb_2d = torch.cat((h_embeds, w_embeds), dim=-1) # Shape: (grid_h, grid_w, head_dim)
        
        # Flatten to a 1D sequence of patches
        emb_flat = emb_2d.reshape(-1, self.head_dim) # Shape: (grid_h * grid_w, head_dim)
        
        # 6. Register the final cos and sin caches
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
        
        # Copy the weight matrices directly to preserve pre-trained knowledge
        self.k_proj = original_attention_module.k_proj
        self.v_proj = original_attention_module.v_proj
        self.q_proj = original_attention_module.q_proj
        self.out_proj = original_attention_module.out_proj
        
        # Instantiate our new 2D RoPE module
        self.rope_2d = RoPE2DEmbedding(self.head_dim, grid_dims)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, attention_mask=None, causal_attention_mask=None, output_attentions=False):
        bsz, seq_len, embed_dim = hidden_states.size()

        # Get Q, K, V projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = self._shape(query_states, seq_len, bsz)
        key_states = self._shape(key_states, seq_len, bsz)
        value_states = self._shape(value_states, seq_len, bsz)

        # --- CORE MODIFICATION: Apply 2D RoPE ---
        # Get the pre-computed cos/sin cache
        cos_sin_cache = self.rope_2d()
        # Apply rotations to query and key
        query_states, key_states = apply_2d_rope(query_states, key_states, cos_sin_cache)
        # --- END OF MODIFICATION ---

        # Standard attention calculation
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
        
        return (attn_output, attn_weights) if output_attentions else (attn_output,)

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

    # --- 1. Disable original positional embeddings using the provided path ---
    try:
        pos_embed_module = get_module_by_name(encoder, positional_embedding_path)
        
        # The actual parameter is often a child of the module (e.g., .weight)
        if hasattr(pos_embed_module, 'weight'):
            pos_embed_module.weight.data.zero_()
            pos_embed_module.weight.requires_grad = False
            logging.info(f"Successfully disabled positional embeddings at '{positional_embedding_path}'.")
        else:
            logging.warning(f"Module at '{positional_embedding_path}' found, but it has no '.weight' attribute to disable.")

    except AttributeError:
        logging.error(f"Could not find the positional embedding module at the specified path: '{positional_embedding_path}'. Conversion might fail or be incorrect.")
        # Depending on strictness, you might want to raise an error here.
        raise AttributeError(f"Path '{positional_embedding_path}' not found in encoder.")

    # --- 2. Recursively find and replace attention modules ---
    # We define a recursive helper function to traverse the model tree.
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
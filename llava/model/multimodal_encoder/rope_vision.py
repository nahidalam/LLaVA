import torch
from torch import nn

# Copied from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
class VisionRotaryEmbedding(nn.Module):
    """The rotary embedding for vision transformer."""
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

# Copied from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    print(f"[DEBUG rope_vision] Inside apply_rotary_pos_emb_vision:")
    print(f"  - q.shape: {q.shape}")
    print(f"  - k.shape: {k.shape}")
    print(f"  - cos.shape: {cos.shape}")
    print(f"  - sin.shape: {sin.shape}")

    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    # The unsqueeze is for the head dimension
    # cos, sin = cos.unsqueeze(1).float(), sin.unsqueeze(1).float()
    # The original unsqueeze was incorrect for the (batch, heads, seq, dim) layout.
    # We need to add dimensions for batch and heads to make it broadcastable.
    # Shape changes from (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0).float()
    sin = sin.unsqueeze(0).unsqueeze(0).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed
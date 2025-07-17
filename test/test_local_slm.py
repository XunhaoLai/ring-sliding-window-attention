import torch
from typing import Optional
from einops import einsum
from ring_swa.streaming_llm.naive import naive_streaming_llm_attn


def torch_streaming_llm_attn_nonvarlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    softmax_scale: Optional[float] = None,
):
    batch_size, q_len, num_q_heads, head_dim = q.shape
    _, k_len, num_kv_heads, _ = k.shape

    # Default softmax scale
    if softmax_scale is None:
        softmax_scale = head_dim ** (-0.5)

    # Handle Grouped-Query Attention (GQA) by repeating K and V heads
    if num_q_heads != num_kv_heads:
        num_shared_heads = num_q_heads // num_kv_heads
        assert (
            num_q_heads % num_kv_heads == 0
        ), "Number of query heads must be divisible by number of key/value heads."
        k = k.repeat_interleave(num_shared_heads, dim=2)
        v = v.repeat_interleave(num_shared_heads, dim=2)

    # Calculate attention scores (scaled dot product)
    # qk shape: (batch_size, num_q_heads, q_len, k_len)
    qk = einsum(q, k, "b i h d, b j h d -> b h i j") * softmax_scale

    # Apply the sliding window mask if window_size is not (-1, -1)
    # Create indices for the query and key sequence dimensions
    q_indices = torch.arange(q_len, device=q.device, dtype=torch.long)
    k_indices = torch.arange(k_len, device=q.device, dtype=torch.long)

    # Create a matrix of relative indices (j - i)
    # Shape: (q_len, k_len)
    relative_indices = k_indices.unsqueeze(0) - q_indices.unsqueeze(1)

    # Calculate the alignment offset for different sequence lengths
    offset = k_len - q_len

    # Initialize masks for left and right windows
    lower_mask = torch.zeros_like(relative_indices, dtype=torch.bool)
    upper_mask = torch.zeros_like(relative_indices, dtype=torch.bool)

    # Calculate the left window boundary
    lower_bound = offset - window_size
    lower_mask = relative_indices < lower_bound

    # Calculate the right window boundary
    upper_bound = offset
    upper_mask = relative_indices > upper_bound

    # Combine masks: True for positions to be masked out
    mask = lower_mask | upper_mask

    # sink token
    mask = ~mask
    mask[:, :1] = True

    # Apply the mask to the attention scores.
    # The mask of shape (q_len, k_len) is broadcast across the batch and head dimensions.
    qk = qk.masked_fill(~mask[None, None, ...], -torch.inf)

    # Apply softmax and compute the final output
    qk = qk.softmax(dim=-1, dtype=torch.float32)
    out = einsum(qk.to(v.dtype), v, "b h i j, b j h d -> b i h d")

    return out


if __name__ == "__main__":
    torch.manual_seed(42)
    seq_len_q = 1024
    seq_len_k = 1024
    window_size = 128

    q = torch.randn(1, seq_len_q, 8, 64).cuda().bfloat16().requires_grad_()
    k = torch.randn(1, seq_len_k, 2, 64).cuda().bfloat16().requires_grad_()
    v = torch.randn(1, seq_len_k, 2, 64).cuda().bfloat16().requires_grad_()

    out = torch_streaming_llm_attn_nonvarlen(q, k, v, window_size)

    q1 = q.detach().clone().requires_grad_()
    k1 = k.detach().clone().requires_grad_()
    v1 = v.detach().clone().requires_grad_()
    out_flash = naive_streaming_llm_attn(q1, k1, v1, window_size)

    try:
        torch.testing.assert_close(out, out_flash, atol=1e-2, rtol=1e-2)
        print("==> forward pass passed")
    except Exception as e:
        print("==> forward pass failed")
        print(e)

    out.backward(torch.ones_like(out))
    out_flash.backward(torch.ones_like(out_flash))

    try:
        torch.testing.assert_close(q1.grad, q.grad, atol=1e-2, rtol=1e-2)
        print("==> backward pass q gradients passed")
    except Exception as e:
        print("==> backward pass q gradients failed")
        print(e)

    try:
        torch.testing.assert_close(k1.grad, k.grad, atol=1e-2, rtol=1e-2)
        print("==> backward pass k gradients passed")
    except Exception as e:
        print("==> backward pass k gradients failed")
        print(e)

    try:
        torch.testing.assert_close(v1.grad, v.grad, atol=1e-2, rtol=1e-2)
        print("==> backward pass v gradients passed")
    except Exception as e:
        print("==> backward pass v gradients failed")
        print(e)

import torch
from typing import Tuple, Optional
from einops import einsum
from flash_attn import flash_attn_func


def torch_sliding_window_attn_nonvarlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: Tuple[int, int],
    softmax_scale: Optional[float] = None,
):
    """
    Computes sliding window attention for variable sequence lengths, mimicking
    the behavior of FlashAttention.

    This function supports cases where the query sequence length is different
    from the key/value sequence length.

    Args:
        q: Query tensor of shape (batch_size, q_len, num_q_heads, head_dim).
        k: Key tensor of shape (batch_size, k_len, num_kv_heads, head_dim).
        v: Value tensor of shape (batch_size, k_len, num_kv_heads, head_dim).
        window_size: A tuple (left_context, right_context). A query at position `i`
                     attends to keys in a window relative to its aligned position.
                     A value of -1 indicates an infinite window in that direction.
        softmax_scale: An optional scaling factor for the softmax function.
                       If None, it defaults to head_dim**(-0.5).

    Returns:
        The output tensor after applying sliding window attention, with shape
        (batch_size, q_len, num_q_heads, head_dim).
    """
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
    if window_size != (-1, -1):
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
        if window_size[0] != -1:
            assert window_size[0] >= 0, "Left window size must be non-negative or -1."
            lower_bound = offset - window_size[0]
            lower_mask = relative_indices < lower_bound

        # Calculate the right window boundary
        if window_size[1] != -1:
            assert window_size[1] >= 0, "Right window size must be non-negative or -1."
            upper_bound = offset + window_size[1]
            upper_mask = relative_indices > upper_bound

        # Combine masks: True for positions to be masked out
        mask = lower_mask | upper_mask

        # Apply the mask to the attention scores.
        # The mask of shape (q_len, k_len) is broadcast across the batch and head dimensions.
        qk = qk.masked_fill(mask[None, None, ...], -torch.inf)

    # Apply softmax and compute the final output
    qk = qk.softmax(dim=-1, dtype=torch.float32)
    out = einsum(qk.to(v.dtype), v, "b h i j, b j h d -> b i h d")

    return out


if __name__ == "__main__":
    torch.manual_seed(42)
    seq_len_q = 100
    seq_len_k = 100
    window_size = (50, 50)

    q = torch.randn(1, seq_len_q, 8, 64).cuda().bfloat16().requires_grad_()
    k = torch.randn(1, seq_len_k, 2, 64).cuda().bfloat16().requires_grad_()
    v = torch.randn(1, seq_len_k, 2, 64).cuda().bfloat16().requires_grad_()

    out = torch_sliding_window_attn_nonvarlen(q, k, v, window_size)

    q1 = q.detach().clone().requires_grad_()
    k1 = k.detach().clone().requires_grad_()
    v1 = v.detach().clone().requires_grad_()
    out_flash = flash_attn_func(q1, k1, v1, window_size=window_size)

    torch.testing.assert_close(out, out_flash, atol=1e-2, rtol=1e-2)
    print("forward pass passed")

    out.backward(torch.ones_like(out))
    out_flash.backward(torch.ones_like(out_flash))

    torch.testing.assert_close(q1.grad, q.grad, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k1.grad, k.grad, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(v1.grad, v.grad, atol=1e-2, rtol=1e-2)
    print("backward pass gradients checked")

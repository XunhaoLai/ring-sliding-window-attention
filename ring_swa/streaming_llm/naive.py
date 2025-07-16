import torch
from ring_swa.ops import merge_attn_out
from ring_swa.ops.wrap_flash import wrapped_flash_attn_fwd, wrapped_flash_attn_bwd


class StreamingLLMAttn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        window_size: int,
        softmax_scale: float,
    ):
        o_swa, lse_swa = wrapped_flash_attn_fwd(
            q[:, 1:],
            k[:, 1:],
            v[:, 1:],
            softmax_scale,
            window_size=(window_size, 0),
            causal=True,
        )
        o_sink, lse_sink = wrapped_flash_attn_fwd(
            q,
            k[:, :1],
            v[:, :1],
            softmax_scale,
            window_size=(-1, -1),
            causal=False,
        )
        o = torch.nn.functional.pad(o_swa, (0, 0, 0, 0, 1, 0), value=0)
        lse = torch.nn.functional.pad(lse_swa, (1, 0), value=-torch.inf)
        merge_attn_out(o, lse, o_sink, lse_sink)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.softmax_scale = softmax_scale
        ctx.window_size = window_size
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor, *args):
        q, k, v, o, lse = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale
        window_size = ctx.window_size
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        wrapped_flash_attn_bwd(
            q[:, 1:],
            k[:, 1:],
            v[:, 1:],
            o[:, 1:],
            lse[:, :, 1:].clone(),
            do[:, 1:],
            dq[:, 1:],
            dk[:, 1:],
            dv[:, 1:],
            softmax_scale,
            window_size=(window_size, 0),
            causal=True,
        )
        dq_sink = torch.zeros_like(q)
        dk_sink = torch.zeros_like(k[:, :1])
        dv_sink = torch.zeros_like(v[:, :1])
        wrapped_flash_attn_bwd(
            q,
            k[:, :1],
            v[:, :1],
            o,
            lse,
            do,
            dq_sink,
            dk_sink,
            dv_sink,
            softmax_scale,
            window_size=(-1, -1),
            causal=False,
        )
        dq.add_(dq_sink)
        dk[:, :1].add_(dk_sink)
        dv[:, :1].add_(dv_sink)
        return dq, dk, dv, None, None


def naive_streaming_llm_attn(q, k, v, window_size, softmax_scale=None):
    """Naive implementation of streaming LLM attention.
    Each query attends to the key and value in the sliding window and the first sink token.

    Args:
        q (torch.Tensor): Query tensor of shape (batch_size, seq_len, num_q_heads, head_dim)
        k (torch.Tensor): Key tensor of shape (batch_size, seq_len, num_kv_heads, head_dim)
        v (torch.Tensor): Value tensor of shape (batch_size, seq_len, num_kv_heads, head_dim)
        window_size (int): Window size for the sliding window attention
        softmax_scale (float, optional): Softmax scale. Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, num_q_heads, head_dim)
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    return StreamingLLMAttn.apply(q, k, v, window_size, softmax_scale)


def naive_streaming_llm_attn_varlen(
    q, k, v, window_size, cu_seqlens, softmax_scale=None
):
    """Naive implementation of streaming LLM attention with variable length.
    Each query attends to the key and value in the sliding window and the first sink token.
    This is a naive implementation that use for loop to handle the variable length.

    Args:
        q (torch.Tensor): Query tensor of shape (batch_size, seq_len, num_q_heads, head_dim)
        k (torch.Tensor): Key tensor of shape (batch_size, seq_len, num_kv_heads, head_dim)
        v (torch.Tensor): Value tensor of shape (batch_size, seq_len, num_kv_heads, head_dim)
        window_size (int): Window size for the sliding window attention
        cu_seqlens (torch.Tensor): Cumulative sequence lengths of shape (batch_size + 1)
        softmax_scale (float, optional): Softmax scale. Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, num_q_heads, head_dim)
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    batch_size = cu_seqlens.shape[0] - 1
    o = []
    for i in range(batch_size):
        q_local = q[cu_seqlens[i] : cu_seqlens[i + 1]]
        k_local = k[cu_seqlens[i] : cu_seqlens[i + 1]]
        v_local = v[cu_seqlens[i] : cu_seqlens[i + 1]]
        o_local = naive_streaming_llm_attn(
            q_local.unsqueeze(0),
            k_local.unsqueeze(0),
            v_local.unsqueeze(0),
            window_size,
            softmax_scale,
        )
        o.append(o_local.squeeze(0))
    o = torch.cat(o, dim=0)
    return o

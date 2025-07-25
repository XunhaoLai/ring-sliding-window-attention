import torch
from ring_swa.ops import merge_attn_out
from ring_swa.ops.wrap_flash import (
    wrapped_flash_attn_fwd,
    wrapped_flash_attn_bwd,
    wrapped_flash_attn_fwd_varlen,
    wrapped_flash_attn_bwd_varlen,
)


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


class StreamingLLMAttnVarlen(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        window_size: int,
        cu_seqlens: torch.Tensor,
        softmax_scale: float,
    ):
        cu_seqlens_no_sink = cu_seqlens.repeat_interleave(2, dim=0)[:-1]
        cu_seqlens_no_sink[1::2] += 1
        max_seqlen_no_sink = (cu_seqlens_no_sink[1:] - cu_seqlens_no_sink[:-1]).max()
        o, lse = wrapped_flash_attn_fwd_varlen(
            q,
            k,
            v,
            softmax_scale,
            window_size=(window_size, 0),
            cu_seqlens_q=cu_seqlens_no_sink,
            cu_seqlens_k=cu_seqlens_no_sink,
            max_seqlen_q=max_seqlen_no_sink,
            max_seqlen_k=max_seqlen_no_sink,
            causal=True,
        )
        k_sink = k[:1].clone()
        v_sink = v[:1].clone()
        o_sink, lse_sink = wrapped_flash_attn_fwd(
            q.unsqueeze(0),
            k_sink.unsqueeze(0),
            v_sink.unsqueeze(0),
            softmax_scale,
            window_size=(-1, -1),
            causal=False,
        )
        o = o.unsqueeze(0)
        lse = lse.unsqueeze(0)
        merge_attn_out(o, lse, o_sink, lse_sink)
        o = o.squeeze(0)
        lse = lse.squeeze(0)
        ctx.save_for_backward(
            q, k, v, o, lse, k_sink, v_sink, cu_seqlens, cu_seqlens_no_sink
        )
        ctx.softmax_scale = softmax_scale
        ctx.window_size = window_size
        ctx.max_seqlen_no_sink = max_seqlen_no_sink
        ctx.max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor, *args):
        (
            q,
            k,
            v,
            o,
            lse,
            k_sink,
            v_sink,
            cu_seqlens,
            cu_seqlens_no_sink,
        ) = ctx.saved_tensors
        batch_size = cu_seqlens.shape[0] - 1
        num_kv_heads, head_dim = k.shape[1:]
        softmax_scale = ctx.softmax_scale
        window_size = ctx.window_size
        max_seqlen_no_sink = ctx.max_seqlen_no_sink
        max_seqlen = ctx.max_seqlen
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        wrapped_flash_attn_bwd_varlen(
            q,
            k,
            v,
            o,
            lse,
            do,
            dq,
            dk,
            dv,
            softmax_scale,
            window_size=(window_size, 0),
            cu_seqlens_q=cu_seqlens_no_sink,
            cu_seqlens_k=cu_seqlens_no_sink,
            max_seqlen_q=max_seqlen_no_sink,
            max_seqlen_k=max_seqlen_no_sink,
            causal=True,
        )
        dq_sink = torch.zeros_like(q)
        k_sink = k_sink.expand(batch_size, num_kv_heads, head_dim)
        v_sink = v_sink.expand(batch_size, num_kv_heads, head_dim)
        dk_sink = torch.zeros_like(k_sink).contiguous()
        dv_sink = torch.zeros_like(v_sink).contiguous()
        cu_seqlens_sink = torch.arange(
            0, batch_size + 1, device=q.device, dtype=torch.int32
        )
        max_seqlen_sink = 1
        wrapped_flash_attn_bwd_varlen(
            q,
            k_sink,
            v_sink,
            o,
            lse,
            do,
            dq_sink,
            dk_sink,
            dv_sink,
            softmax_scale,
            window_size=(-1, -1),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens_sink,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen_sink,
            causal=False,
        )
        dq.add_(dq_sink)
        dk[cu_seqlens[:-1]] += dk_sink
        dv[cu_seqlens[:-1]] += dv_sink
        return dq, dk, dv, None, None, None


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
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    return StreamingLLMAttnVarlen.apply(q, k, v, window_size, cu_seqlens, softmax_scale)

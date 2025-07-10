from typing import Tuple
import torch
from .wrap_flash import (
    wrapped_flash_attn_fwd,
    wrapped_flash_attn_fwd_varlen,
    wrapped_flash_attn_bwd,
    wrapped_flash_attn_bwd_varlen,
)
from .merge import merge_attn_out


def flash_attn_fwd_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    softmax_scale: float,
    window_size: Tuple[int, int],
    causal: bool,
    global_attn_out: torch.Tensor,
    global_softmax_lse: torch.Tensor,
):
    if q.shape[1] <= kv.shape[2]:
        # compute attention
        out, softmax_lse = wrapped_flash_attn_fwd(
            q=q,
            k=kv[0],
            v=kv[1],
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )
        # correct output and lse
        merge_attn_out(
            global_attn_out,
            global_softmax_lse,
            out,
            softmax_lse,
        )
    else:
        # only to handle the case that query length is larger than key length, and window_size[0] >=0, causal=False
        kv_len = kv.shape[2]
        # compute full part
        flash_attn_fwd_func(
            q[:, :-kv_len],
            kv,
            softmax_scale,
            window_size=(-1, -1),
            causal=False,
            global_attn_out=global_attn_out[:, :-kv_len],
            global_softmax_lse=global_softmax_lse[:, :, :-kv_len],
        )
        # compute window part
        flash_attn_fwd_func(
            q[:, -kv_len:],
            kv,
            softmax_scale,
            window_size=window_size,
            causal=False,
            global_attn_out=global_attn_out[:, -kv_len:],
            global_softmax_lse=global_softmax_lse[:, :, -kv_len:],
        )


def flash_attn_bwd_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,
    do: torch.Tensor,
    dq: torch.Tensor,
    dkv: torch.Tensor,
    softmax_scale: float,
    window_size: Tuple[int, int],
    causal: bool,
) -> None:
    if q.shape[1] <= kv.shape[2]:
        wrapped_flash_attn_bwd(
            q=q,
            k=kv[0],
            v=kv[1],
            o=o,
            softmax_lse=softmax_lse,
            do=do,
            dq=dq,
            dk=dkv[0],
            dv=dkv[1],
            softmax_scale=softmax_scale,
            window_size=window_size,
            causal=causal,
        )
    else:
        # only to handle the case that query length is larger than key length, and window_size[0] >=0, causal=False
        kv_len = kv.shape[2]
        dkv_accum = torch.zeros_like(dkv, dtype=torch.float32)
        # compute full part
        flash_attn_bwd_func(
            q[:, :-kv_len],
            kv,
            o[:, :-kv_len],
            softmax_lse[
                :, :, :-kv_len
            ].clone(),  # FIXME: avoid bugs at backward, don't know why
            do[:, :-kv_len],
            dq[:, :-kv_len],
            dkv,
            softmax_scale,
            window_size=(-1, -1),
            causal=causal,
        )
        dkv_accum.add_(dkv)
        # compute window part
        flash_attn_bwd_func(
            q[:, -kv_len:],
            kv,
            o[:, -kv_len:],
            softmax_lse[
                :, :, -kv_len:
            ].clone(),  # FIXME: avoid bugs at backward, don't know why
            do[:, -kv_len:],
            dq[:, -kv_len:],
            dkv,
            softmax_scale,
            window_size=window_size,
            causal=causal,
        )
        dkv_accum.add_(dkv)
        dkv.copy_(dkv_accum.to(dkv.dtype))


def flash_attn_varlen_fwd_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    softmax_scale: float,
    window_size: Tuple[int, int],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: torch.Tensor,
    max_seqlen_k: torch.Tensor,
    causal: bool,
    global_attn_out: torch.Tensor,
    global_softmax_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out, softmax_lse = wrapped_flash_attn_fwd_varlen(
        q=q.squeeze(0),
        k=kv[0].squeeze(0),
        v=kv[1].squeeze(0),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
    )
    # correct output and lse
    merge_attn_out(
        global_attn_out,
        global_softmax_lse,
        out.unsqueeze(0),
        softmax_lse.unsqueeze(0),
    )


def flash_attn_varlen_bwd_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,
    do: torch.Tensor,
    dq: torch.Tensor,
    dkv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: torch.Tensor,
    max_seqlen_k: torch.Tensor,
    softmax_scale: float,
    window_size: Tuple[int, int],
    causal: bool,
) -> None:
    wrapped_flash_attn_bwd_varlen(
        q=q.squeeze(0),
        k=kv[0].squeeze(0),
        v=kv[1].squeeze(0),
        o=o.squeeze(0),
        softmax_lse=softmax_lse.squeeze(0),
        do=do.squeeze(0),
        dq=dq.squeeze(0),
        dk=dkv[0].squeeze(0),
        dv=dkv[1].squeeze(0),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
    )

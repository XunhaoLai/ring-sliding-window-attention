import os

import triton
import triton.language as tl
from packaging.version import parse as parse_version
from importlib.metadata import version
from flash_attn.flash_attn_interface import (
    _flash_attn_backward,
    _flash_attn_forward,
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)

try:
    from flash_attn_interface import (
        _flash_attn_forward as _flash_attn_3_forward,
        _flash_attn_backward as _flash_attn_3_backward,
    )

    FLASH_ATTENTION_3 = os.getenv("FLASH_ATTENTION_3", "0") == "1"
    if FLASH_ATTENTION_3:
        print("[RingSWA] Flash Attention 3 is installed and enabled")
    else:
        print(
            "[RingSWA] Flash Attention 3 is installed but not enabled, use Flash Attention 2"
        )
except Exception:
    FLASH_ATTENTION_3 = False
    _flash_attn_3_forward = None
    _flash_attn_3_backward = None
    print("[RingSWA] Flash Attention 3 is not installed, use Flash Attention 2")

try:
    flash_attn_version = parse_version(version("flash-attn"))
except Exception:
    # Fallback if the package is not found or version is not parsable
    flash_attn_version = parse_version("0.0.0")
    raise ImportError(
        "[RingSWA] flash-attn is not installed! Please install it before using ring_swa."
    )

if flash_attn_version < parse_version("2.5.0"):
    raise ImportError(
        f"[RingSWA] flash-attn version {flash_attn_version} is too old! Please update it to 2.5.0 or higher before using ring_swa."
    )

kwargs_flash_attn_fwd = {
    "dropout_p": 0.0,
    "alibi_slopes": None,
    "return_softmax": False,
}
kwargs_flash_attn_fwd_varlen = {
    "dropout_p": 0.0,
    "alibi_slopes": None,
    "return_softmax": False,
    "block_table": None,
}
kwargs_flash_attn_bwd = {
    "dropout_p": 0.0,
    "alibi_slopes": None,
    "deterministic": False,
    "rng_state": None,
}
kwargs_flash_attn_bwd_varlen = {
    "dropout_p": 0.0,
    "alibi_slopes": None,
    "deterministic": False,
    "rng_state": None,
}
kwargs_flash_attn_3_fwd = {
    "k_new": None,
    "v_new": None,
    "qv": None,
    "out": None,
    "cu_seqlens_q": None,
    "cu_seqlens_k": None,
    "cu_seqlens_k_new": None,
    "seqused_q": None,
    "seqused_k": None,
    "max_seqlen_q": None,
    "max_seqlen_k": None,
    "page_table": None,
    "kv_batch_idx": None,
    "leftpad_k": None,
    "rotary_cos": None,
    "rotary_sin": None,
    "seqlens_rotary": None,
    "q_descale": None,
    "k_descale": None,
    "v_descale": None,
}
kwargs_flash_attn_3_bwd = {
    "cu_seqlens_q": None,
    "cu_seqlens_k": None,
    "sequed_q": None,
    "sequed_k": None,
    "max_seqlen_q": None,
    "max_seqlen_k": None,
}
kwargs_flash_attn_3_fwd_varlen = {
    "k_new": None,
    "v_new": None,
    "qv": None,
    "out": None,
    "cu_seqlens_k_new": None,
    "seqused_q": None,
    "seqused_k": None,
    "page_table": None,
    "kv_batch_idx": None,
    "leftpad_k": None,
    "rotary_cos": None,
    "rotary_sin": None,
    "seqlens_rotary": None,
    "q_descale": None,
    "k_descale": None,
    "v_descale": None,
}
kwargs_flash_attn_3_bwd_varlen = {
    "sequed_q": None,
    "sequed_k": None,
}


if flash_attn_version >= parse_version("2.6.0"):
    kwargs_flash_attn_fwd.update(
        {
            "softcap": False,
        }
    )
    kwargs_flash_attn_fwd_varlen.update(
        {
            "softcap": False,
        }
    )
    kwargs_flash_attn_bwd.update(
        {
            "softcap": False,
        }
    )
    kwargs_flash_attn_bwd_varlen.update(
        {
            "softcap": False,
        }
    )


if flash_attn_version >= parse_version("2.7.0"):
    kwargs_flash_attn_fwd_varlen.update(
        {
            "leftpad_k": None,
            "seqused_k": None,
        }
    )

if flash_attn_version >= parse_version("2.7.4"):
    kwargs_flash_attn_fwd_varlen.update(
        {
            "zero_tensors": False,
        }
    )
    kwargs_flash_attn_bwd_varlen.update(
        {
            "zero_tensors": False,
        }
    )


def wrapped_flash_attn_fwd(q, k, v, softmax_scale, window_size, causal):
    if FLASH_ATTENTION_3:
        out, softmax_lse, *rest = _flash_attn_3_forward(
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            **kwargs_flash_attn_3_fwd,
        )
    elif flash_attn_version < parse_version("2.7.0"):
        _, _, _, _, out, softmax_lse, _, _ = _flash_attn_forward(
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            **kwargs_flash_attn_fwd,
        )
    else:
        out, softmax_lse, _, _ = _flash_attn_forward(
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            **kwargs_flash_attn_fwd,
        )
    return out, softmax_lse


def wrapped_flash_attn_fwd_varlen(
    q,
    k,
    v,
    softmax_scale,
    window_size,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    causal,
):
    if FLASH_ATTENTION_3:
        out, softmax_lse, *rest = _flash_attn_3_forward(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            **kwargs_flash_attn_3_fwd_varlen,
        )
    elif flash_attn_version < parse_version("2.6.0"):
        _, _, _, _, out, softmax_lse, _, _ = _flash_attn_varlen_forward(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            **kwargs_flash_attn_fwd_varlen,
        )
        softmax_lse = rmpad_softmax_lse(softmax_lse, cu_seqlens_q)
    elif flash_attn_version < parse_version("2.7.0"):
        _, _, _, _, out, softmax_lse, _, _ = _flash_attn_varlen_forward(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            **kwargs_flash_attn_fwd_varlen,
        )
    else:
        out, softmax_lse, _, _ = _flash_attn_varlen_forward(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            **kwargs_flash_attn_fwd_varlen,
        )
    return out, softmax_lse


def wrapped_flash_attn_bwd(
    q, k, v, o, softmax_lse, do, dq, dk, dv, softmax_scale, window_size, causal
):
    if FLASH_ATTENTION_3:
        _flash_attn_3_backward(
            dout=do,
            q=q,
            k=k,
            v=v,
            out=o,
            softmax_lse=softmax_lse,
            dq=dq,
            dk=dk,
            dv=dv,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            **kwargs_flash_attn_3_bwd,
        )
    elif flash_attn_version < parse_version("2.7.0"):
        _flash_attn_backward(
            dout=do,
            q=q,
            k=k,
            v=v,
            out=o,
            softmax_lse=softmax_lse,
            dq=dq,
            dk=dk,
            dv=dv,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            **kwargs_flash_attn_bwd,
        )
    else:
        _flash_attn_backward(
            dout=do,
            q=q,
            k=k,
            v=v,
            out=o,
            softmax_lse=softmax_lse,
            dq=dq,
            dk=dk,
            dv=dv,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            **kwargs_flash_attn_bwd,
        )


def wrapped_flash_attn_bwd_varlen(
    q,
    k,
    v,
    o,
    softmax_lse,
    do,
    dq,
    dk,
    dv,
    softmax_scale,
    window_size,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    causal,
):
    if FLASH_ATTENTION_3:
        _flash_attn_3_backward(
            dout=do,
            q=q,
            k=k,
            v=v,
            out=o,
            softmax_lse=softmax_lse,
            dq=dq,
            dk=dk,
            dv=dv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            **kwargs_flash_attn_3_bwd_varlen,
        )
    elif flash_attn_version < parse_version("2.6.0"):
        softmax_lse = pad_softmax_lse(softmax_lse, cu_seqlens_q)
        _flash_attn_varlen_backward(
            dout=do,
            q=q,
            k=k,
            v=v,
            out=o,
            softmax_lse=softmax_lse,
            dq=dq,
            dk=dk,
            dv=dv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            **kwargs_flash_attn_bwd_varlen,
        )
    elif flash_attn_version < parse_version("2.7.0"):
        _flash_attn_varlen_backward(
            dout=do,
            q=q,
            k=k,
            v=v,
            out=o,
            softmax_lse=softmax_lse,
            dq=dq,
            dk=dk,
            dv=dv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            **kwargs_flash_attn_bwd_varlen,
        )
    else:
        _flash_attn_varlen_backward(
            dout=do,
            q=q,
            k=k,
            v=v,
            out=o,
            softmax_lse=softmax_lse,
            dq=dq,
            dk=dk,
            dv=dv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            **kwargs_flash_attn_bwd,
        )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_N": BN, "BLOCK_SIZE_H": BH}, num_warps=nw, num_stages=ns
        )
        for BN in [64, 128, 256]
        for BH in [16, 32, 64]
        for nw in [2, 4]
        for ns in [1, 2, 3]
    ],
    key=["num_heads"],
)
@triton.jit(do_not_specialize=["max_seqlen"])
def _rmpad_softmax_lse_kernel(
    old_lse_ptr,
    new_lse_ptr,
    cu_seqlens,
    batch_size,
    num_heads,
    max_seqlen,
    stride_ol_b,
    stride_ol_h,
    stride_ol_n,
    stride_nl_h,
    stride_nl_n,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b, pid_h, pid_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # get q k start and len after rmpad
    seq_start = tl.load(cu_seqlens + pid_b)
    seq_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    if BLOCK_SIZE_N * pid_n >= seq_len:
        return
    old_lse_ptrs = tl.make_block_ptr(
        base=old_lse_ptr + pid_b * stride_ol_b,
        shape=(num_heads, max_seqlen),
        strides=(stride_ol_h, stride_ol_n),
        offsets=(pid_h * BLOCK_SIZE_H, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_N),
        order=(1, 0),
    )
    new_lse_ptrs = tl.make_block_ptr(
        base=new_lse_ptr + seq_start * stride_nl_n,
        shape=(num_heads, seq_len),
        strides=(stride_nl_h, stride_nl_n),
        offsets=(pid_h * BLOCK_SIZE_H, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_N),
        order=(1, 0),
    )
    old_lse = tl.load(old_lse_ptrs, boundary_check=(0, 1))
    tl.store(
        new_lse_ptrs, old_lse.to(new_lse_ptrs.dtype.element_ty), boundary_check=(0, 1)
    )


def rmpad_softmax_lse(softmax_lse, cu_seqlens):
    batch_size, num_heads, max_seqlen = softmax_lse.shape
    assert batch_size == cu_seqlens.shape[0] - 1
    new_lse = softmax_lse.new_empty(num_heads, cu_seqlens[-1])

    def grid(META):
        return (
            batch_size,
            triton.cdiv(num_heads, META["BLOCK_SIZE_H"]),
            triton.cdiv(max_seqlen, META["BLOCK_SIZE_N"]),
        )

    _rmpad_softmax_lse_kernel[grid](
        softmax_lse,
        new_lse,
        cu_seqlens,
        batch_size,
        num_heads,
        max_seqlen,
        softmax_lse.stride(0),
        softmax_lse.stride(1),
        softmax_lse.stride(2),
        new_lse.stride(0),
        new_lse.stride(1),
    )

    return new_lse


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_N": BN, "BLOCK_SIZE_H": BH}, num_warps=nw, num_stages=ns
        )
        for BN in [64, 128, 256]
        for BH in [16, 32, 64]
        for nw in [2, 4]
        for ns in [1, 2, 3]
    ],
    key=["num_heads"],
    reset_to_zero=["new_lse_ptr"],
)
@triton.jit(do_not_specialize=["max_seqlen"])
def _pad_softmax_lse_kernel(
    old_lse_ptr,
    new_lse_ptr,
    cu_seqlens,
    batch_size,
    num_heads,
    max_seqlen,
    stride_ol_h,
    stride_ol_n,
    stride_nl_b,
    stride_nl_h,
    stride_nl_n,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b, pid_h, pid_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # get q k start and len after rmpad
    seq_start = tl.load(cu_seqlens + pid_b)
    seq_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    if BLOCK_SIZE_N * pid_n >= seq_len:
        return
    old_lse_ptrs = tl.make_block_ptr(
        base=old_lse_ptr + seq_start * stride_ol_n,
        shape=(num_heads, seq_len),
        strides=(stride_ol_h, stride_ol_n),
        offsets=(pid_h * BLOCK_SIZE_H, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_N),
        order=(1, 0),
    )
    new_lse_ptrs = tl.make_block_ptr(
        base=new_lse_ptr + pid_b * stride_nl_b,
        shape=(num_heads, max_seqlen),
        strides=(stride_nl_h, stride_nl_n),
        offsets=(pid_h * BLOCK_SIZE_H, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_N),
        order=(1, 0),
    )
    old_lse = tl.load(old_lse_ptrs, boundary_check=(0, 1))
    tl.store(
        new_lse_ptrs, old_lse.to(new_lse_ptrs.dtype.element_ty), boundary_check=(0, 1)
    )


def pad_softmax_lse(softmax_lse, cu_seqlens):
    num_heads, total_seqlen = softmax_lse.shape
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
    batch_size = cu_seqlens.shape[0] - 1
    if batch_size == 1:
        return softmax_lse.unsqueeze(0)
    new_lse = softmax_lse.new_zeros(batch_size, num_heads, max_seqlen)

    def grid(META):
        return (
            batch_size,
            triton.cdiv(num_heads, META["BLOCK_SIZE_H"]),
            triton.cdiv(max_seqlen, META["BLOCK_SIZE_N"]),
        )

    _pad_softmax_lse_kernel[grid](
        softmax_lse,
        new_lse,
        cu_seqlens,
        batch_size,
        num_heads,
        max_seqlen,
        softmax_lse.stride(0),
        softmax_lse.stride(1),
        new_lse.stride(0),
        new_lse.stride(1),
        new_lse.stride(2),
    )

    return new_lse

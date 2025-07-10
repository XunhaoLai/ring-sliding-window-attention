import math
import torch
from torch.distributed import ProcessGroup
from typing import Optional
from ring_swa.ops import (
    ring_attn_p2p_communicate,
    flash_attn_fwd_func,
    flash_attn_bwd_func,
)
from flash_attn import flash_attn_func


MIN_P2P_COMM_BUFFER_LENGTH = 3


class RingSlidingWindowAttn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        window_size: int,
        cp_group: ProcessGroup,
        cp_size: int,
        cp_local_rank: int,
        cp_prev_global_rank: int,
        cp_next_global_rank: int,
        cp_stream: Optional[torch.cuda.Stream],
        softmax_scale: Optional[float],
    ):
        batch_size, seq_len, num_q_heads, head_dim = q.shape
        batch_size, seq_len, num_kv_heads, head_dim = k.shape
        batch_size, seq_len, num_kv_heads, head_dim = v.shape
        assert (
            q.shape[1] == k.shape[1] == v.shape[1]
        ), "Sequence length must be the same"
        assert (
            num_q_heads % num_kv_heads == 0
        ), "Number of query heads must be divisible by number of key/value heads"
        if softmax_scale is None:
            softmax_scale = head_dim ** (-0.5)

        # communicate times
        ring_comm_times = math.ceil(window_size / seq_len)
        ring_comm_times = min(
            ring_comm_times, cp_size - 1
        )  # max ring comm times is cp_size - 1
        send_rank, recv_rank = cp_next_global_rank, cp_prev_global_rank

        # init out, lse
        attn_out = torch.zeros_like(q, dtype=torch.float32)
        softmax_lse = torch.full(
            (batch_size, num_q_heads, seq_len),
            -torch.inf,
            dtype=torch.double,
            device=q.device,
        )

        # pack key and value for p2p comm, shape [2, batch_size, seq_len, num_kv_heads, head_dim]
        kv = torch.stack([k, v], dim=0)

        # init p2p comm buffers for kv ring send/recv
        send_recv_reqs = []
        p2p_comm_buffers = [kv]
        p2p_comm_buffers_length = min(MIN_P2P_COMM_BUFFER_LENGTH, ring_comm_times + 1)
        for _ in range(p2p_comm_buffers_length - 1):
            p2p_comm_buffers.append(torch.empty_like(kv))

        for i in range(ring_comm_times + 1):
            cur_buffer_idx, nxt_buffer_idx = (
                i % p2p_comm_buffers_length,
                (i + 1) % p2p_comm_buffers_length,
            )

            # wait until kv is received from prev rank
            for req in send_recv_reqs:
                req.wait()
            # get received kv from buffer
            kv = p2p_comm_buffers[cur_buffer_idx]

            # send kv to next rank, don't need to send in the last step
            if i < ring_comm_times:
                send_recv_reqs = ring_attn_p2p_communicate(
                    rank=cp_local_rank,
                    send_tensor=p2p_comm_buffers[cur_buffer_idx],
                    send_rank=send_rank,
                    recv_tensor=p2p_comm_buffers[nxt_buffer_idx],
                    recv_rank=recv_rank,
                    cp_group=cp_group,
                )

            # compute attention
            q_id = cp_local_rank
            kv_id = (cp_local_rank - i) % cp_size
            if q_id < kv_id:  # q will only attend to kv that is ahead of it
                # print(f"[forward] rank: {cp_local_rank}, step: {i}, q_id: {q_id}, kv_id: {kv_id}, skip")
                pass
            elif i == 0:  # first step, compute local block
                # print(f"[forward] rank: {cp_local_rank}, step: {i}, q_id: {q_id}, kv_id: {kv_id}, compute local block")
                flash_attn_fwd_func(
                    q,
                    kv,
                    window_size=(window_size, 0),
                    softmax_scale=softmax_scale,
                    causal=True,
                    global_attn_out=attn_out,
                    global_softmax_lse=softmax_lse,
                )
            elif (i + 1) * seq_len <= window_size + 1:  # full block
                # print(f"[forward] rank: {cp_local_rank}, step: {i}, q_id: {q_id}, kv_id: {kv_id}, full block")
                flash_attn_fwd_func(
                    q,
                    kv,
                    window_size=(-1, -1),
                    softmax_scale=softmax_scale,
                    causal=False,
                    global_attn_out=attn_out,
                    global_softmax_lse=softmax_lse,
                )
            elif (
                (i + 1) * seq_len > window_size + 1 > i * seq_len
            ):  # only skip some triangular part at left down corner
                # print(f"[forward] rank: {cp_local_rank}, step: {i}, q_id: {q_id}, kv_id: {kv_id}, only skip some triangular part at left down corner")
                flash_attn_fwd_func(
                    q,
                    kv,
                    window_size=(window_size % seq_len, seq_len),
                    softmax_scale=softmax_scale,
                    causal=False,
                    global_attn_out=attn_out,
                    global_softmax_lse=softmax_lse,
                )
            elif (
                i * seq_len > window_size > (i - 1) * seq_len
            ):  # only compute some triangular part at right up corner
                # print(f"[forward] rank: {cp_local_rank}, step: {i}, q_id: {q_id}, kv_id: {kv_id}, only compute some triangular part at right up corner")
                valid_len = window_size % seq_len
                flash_attn_fwd_func(
                    q[:, :valid_len],
                    kv[:, :, -valid_len:],
                    window_size=(0, seq_len),
                    softmax_scale=softmax_scale,
                    causal=False,
                    global_attn_out=attn_out[:, :valid_len],
                    global_softmax_lse=softmax_lse[:, :, :valid_len],
                )
            else:
                raise ValueError(f"[forward] Invalid step: {i}")

        # cast dtype
        attn_out = attn_out.to(v.dtype)
        softmax_lse = softmax_lse.to(torch.float32)

        ctx.save_for_backward(q, kv, attn_out, softmax_lse)
        ctx.softmax_scale = softmax_scale
        ctx.window_size = window_size
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.cp_local_rank = cp_local_rank
        ctx.cp_prev_global_rank = cp_prev_global_rank
        ctx.cp_next_global_rank = cp_next_global_rank
        return attn_out

    @staticmethod
    def backward(ctx, do: torch.Tensor, *args):
        # get saved tensors
        q, kv, o, softmax_lse = ctx.saved_tensors
        seq_len = q.shape[1]

        # get args
        softmax_scale = ctx.softmax_scale
        cp_group = ctx.cp_group
        cp_size = ctx.cp_size
        cp_local_rank = ctx.cp_local_rank
        cp_prev_global_rank = ctx.cp_prev_global_rank
        cp_next_global_rank = ctx.cp_next_global_rank
        window_size = ctx.window_size

        # communicate times, and for backward, the ring direction opposite from forward
        ring_comm_times = math.ceil(window_size / seq_len)
        ring_comm_times = min(
            ring_comm_times, cp_size - 1
        )  # max ring comm times is cp_size - 1
        send_rank, recv_rank = cp_prev_global_rank, cp_next_global_rank

        # here kv is not from chunk cp_local_rank it self
        init_kv_id = (cp_local_rank - ring_comm_times + cp_size) % cp_size

        # init global dq, dkv and local dq, dkv
        dq = torch.zeros_like(q, dtype=torch.float32)
        dkv = torch.zeros_like(kv, dtype=torch.float32)
        dq_local = torch.empty_like(q, dtype=q.dtype)
        dkv_local = torch.empty_like(kv, dtype=kv.dtype)

        # init p2p comm buffers for kv ring send/recv
        kv_send_recv_reqs = []
        dkv_send_recv_reqs = []
        p2p_comm_buffers_length = min(MIN_P2P_COMM_BUFFER_LENGTH, ring_comm_times + 1)
        kv_p2p_comm_buffers = [kv]
        dkv_p2p_comm_buffers = [dkv]
        for _ in range(p2p_comm_buffers_length - 1):
            kv_p2p_comm_buffers.append(torch.empty_like(kv))
            dkv_p2p_comm_buffers.append(torch.empty_like(dkv))

        # backward compute
        for i in range(ring_comm_times + 1):
            cur_buffer_idx, nxt_buffer_idx = (
                i % p2p_comm_buffers_length,
                (i + 1) % p2p_comm_buffers_length,
            )

            # wait until kv is received from next rank
            for req in kv_send_recv_reqs:
                req.wait()

            # get received kv from buffer
            kv = kv_p2p_comm_buffers[cur_buffer_idx]

            # send kv to previous rank, don't need to send in the last step
            if i < ring_comm_times:
                kv_send_recv_reqs = ring_attn_p2p_communicate(
                    rank=cp_local_rank,
                    send_tensor=kv_p2p_comm_buffers[cur_buffer_idx],
                    send_rank=send_rank,
                    recv_tensor=kv_p2p_comm_buffers[nxt_buffer_idx],
                    recv_rank=recv_rank,
                    cp_group=cp_group,
                )

            # compute attention backward
            q_id = cp_local_rank
            kv_id = (init_kv_id + i) % cp_size
            fwd_i = ring_comm_times - i
            has_compute = False
            if q_id < kv_id:  # q will only attend to kv that is ahead of it
                # print(f"[backward] rank: {cp_local_rank}, step: {i}, q_id: {q_id}, kv_id: {kv_id}, skip")
                has_compute = False
            elif fwd_i == 0:  # compute local block
                # print(f"[backward] rank: {cp_local_rank}, step: {i}, q_id: {q_id}, kv_id: {kv_id}, compute local block")
                flash_attn_bwd_func(
                    q,
                    kv,
                    o,
                    softmax_lse,
                    do,
                    dq_local,
                    dkv_local,
                    window_size=(window_size, 0),
                    softmax_scale=softmax_scale,
                    causal=True,
                )
                has_compute = True
            elif (fwd_i + 1) * seq_len <= window_size + 1:  # full block
                # print(f"[backward] rank: {cp_local_rank}, step: {i}, q_id: {q_id}, kv_id: {kv_id}, full block")
                flash_attn_bwd_func(
                    q,
                    kv,
                    o,
                    softmax_lse,
                    do,
                    dq_local,
                    dkv_local,
                    window_size=(-1, -1),
                    softmax_scale=softmax_scale,
                    causal=False,
                )
                has_compute = True
            elif (
                (fwd_i + 1) * seq_len > window_size + 1 > fwd_i * seq_len
            ):  # only skip some triangular part at left down corner
                # print(f"[backward] rank: {cp_local_rank}, step: {i}, q_id: {q_id}, kv_id: {kv_id}, only skip some triangular part at left down corner")
                flash_attn_bwd_func(
                    q,
                    kv,
                    o,
                    softmax_lse,
                    do,
                    dq_local,
                    dkv_local,
                    window_size=(window_size % seq_len, seq_len),
                    softmax_scale=softmax_scale,
                    causal=False,
                )
                has_compute = True
            elif (
                fwd_i * seq_len > window_size > (fwd_i - 1) * seq_len
            ):  # only compute some triangular part at right up corner
                # print(f"[backward] rank: {cp_local_rank}, step: {i}, q_id: {q_id}, kv_id: {kv_id}, only compute some triangular part at right up corner")
                valid_len = window_size % seq_len
                dq_local.zero_()
                dkv_local.zero_()
                flash_attn_bwd_func(
                    q[:, :valid_len],
                    kv[:, :, -valid_len:],
                    o[:, :valid_len],
                    softmax_lse[
                        :, :, :valid_len
                    ].clone(),  # FIXME: avoid bugs at backward, don't know why
                    do[:, :valid_len],
                    dq_local[:, :valid_len],
                    dkv_local[:, :, -valid_len:],
                    window_size=(0, seq_len),
                    softmax_scale=softmax_scale,
                    causal=False,
                )
                has_compute = True
            else:
                raise ValueError(f"[backward] Invalid step: {i}")

            # wait until kv is received from next rank
            for req in dkv_send_recv_reqs:
                req.wait()

            # get received kv from buffer
            dkv = dkv_p2p_comm_buffers[cur_buffer_idx]

            # update global dq, dkv
            if has_compute:
                dq.add_(dq_local)
                dkv.add_(dkv_local)

            # send dkv to previous rank, don't need to send in the last step
            if i < ring_comm_times:
                dkv_send_recv_reqs = ring_attn_p2p_communicate(
                    rank=cp_local_rank,
                    send_tensor=dkv_p2p_comm_buffers[cur_buffer_idx],
                    send_rank=send_rank,
                    recv_tensor=dkv_p2p_comm_buffers[nxt_buffer_idx],
                    recv_rank=recv_rank,
                    cp_group=cp_group,
                )

        dq = dq.to(q.dtype)
        dk = dkv[0].to(kv.dtype)
        dv = dkv[1].to(kv.dtype)
        return dq, dk, dv, None, None, None, None, None, None, None, None


def ring_sliding_window_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    cp_group: ProcessGroup,
    cp_size: int,
    cp_local_rank: int,
    cp_prev_global_rank: int,
    cp_next_global_rank: int,
    softmax_scale: Optional[float] = None,
):
    """
    Ring sliding window attention without variable length.

    Args:
        q (torch.Tensor): Query tensor at current rank, shape [batch_size, seq_len, num_q_heads, head_dim].
        k (torch.Tensor): Key tensor at current rank, shape [batch_size, seq_len, num_kv_heads, head_dim].
        v (torch.Tensor): Value tensor at current rank, shape [batch_size, seq_len, num_kv_heads, head_dim].
        window_size (int): Window size, the number of tokens that each query will attend to, similar to window_size[0] in flash_attn.
        cp_group (ProcessGroup): Process group for context parallelism.
        cp_size (int): Number of ranks in the process group.
        cp_local_rank (int): Local rank.
        cp_prev_global_rank (int): Previous global rank.
        cp_next_global_rank (int): Next global rank.
        softmax_scale (Optional[float], optional): Softmax scale. Defaults to None, which is head_dim ** (-0.5).

    Returns:
        torch.Tensor: Attention output at current rank, shape [seq_len, num_q_heads, head_dim]
    """
    # in this implementation, if you set window_size to w, then each query will attend to w + 1 tokens, include the current token
    # this is same as the one in flash attention
    if cp_size == 1:
        return flash_attn_func(
            q=q,
            k=k,
            v=v,
            window_size=(window_size, 0),
            softmax_scale=softmax_scale,
            causal=True,
        )
    else:
        return RingSlidingWindowAttn.apply(
            q,
            k,
            v,
            window_size,
            cp_group,
            cp_size,
            cp_local_rank,
            cp_prev_global_rank,
            cp_next_global_rank,
            None,
            softmax_scale,
        )

import math
import torch
from torch.distributed import ProcessGroup
from typing import Optional
from ring_swa.ops import (
    ring_attn_p2p_communicate,
    flash_attn_fwd_func,
    flash_attn_bwd_func,
    flash_attn_varlen_fwd_func,
    flash_attn_varlen_bwd_func,
    get_group_local_to_global_rank_map,
)
from flash_attn import flash_attn_varlen_func


MIN_P2P_COMM_BUFFER_LENGTH = 3


class RingSlidingWindowAttnVarlen(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        window_size: int,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cp_group: ProcessGroup,
        cp_size: int,
        cp_local_rank: int,
        cp_prev_global_rank: int,
        cp_next_global_rank: int,
        cp_stream: Optional[torch.cuda.Stream],
        softmax_scale: Optional[float],
    ):
        seq_len, num_q_heads, head_dim = q.shape
        seq_len, num_kv_heads, head_dim = k.shape
        seq_len, num_kv_heads, head_dim = v.shape
        assert (
            q.shape[0] == k.shape[0] == v.shape[0]
        ), "Sequence length must be the same"
        assert (
            num_q_heads % num_kv_heads == 0
        ), "Number of query heads must be divisible by number of key/value heads"
        if softmax_scale is None:
            softmax_scale = head_dim ** (-0.5)

        # unsqueeze q, k, v to batch_size = 1
        batch_size = 1
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

        # compute local_cu_seqlens and local_max_len for diag block
        local_seq_start = cp_local_rank * seq_len
        local_seq_end = (
            local_seq_start + seq_len
        )  # FIXME: assume seq_len is the same for all ranks
        first_seq_id = torch.where(cu_seqlens > local_seq_start)[0][0]
        last_seq_id = torch.where(cu_seqlens >= local_seq_end)[0][0]
        local_cu_seqlens = cu_seqlens[first_seq_id:last_seq_id] - local_seq_start
        local_cu_seqlens = torch.cat(
            [
                cu_seqlens.new_tensor([0]),
                local_cu_seqlens,
                cu_seqlens.new_tensor([seq_len]),
            ],
            dim=0,
        )
        local_max_len = (local_cu_seqlens[1:] - local_cu_seqlens[:-1]).max()
        # compute first sequence's length for non-diag block
        first_q_len = local_cu_seqlens[1]
        first_kv_len = cp_local_rank * seq_len - cu_seqlens[first_seq_id - 1]
        # print(f"[forward] rank: {cp_local_rank}, first_q_len: {first_q_len}, first_kv_len: {first_kv_len}")

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

            # qkv id is the id of rank, not the batch id
            q_id = cp_local_rank
            kv_id = (cp_local_rank - i) % cp_size
            if q_id < kv_id:  # q will only attend to kv that is ahead of it
                # print(f"[forward] rank: {cp_local_rank}, step: {i}, q_id: {q_id}, kv_id: {kv_id}, skip")
                pass
            elif i == 0:  # first step, compute local block
                # print(f"[forward] rank: {cp_local_rank}, step: {i}, q_id: {q_id}, kv_id: {kv_id}, compute local block, local_cu_seqlens: {local_cu_seqlens.tolist()}, local_max_len: {local_max_len}")
                flash_attn_varlen_fwd_func(
                    q,
                    kv,
                    cu_seqlens_q=local_cu_seqlens,
                    cu_seqlens_k=local_cu_seqlens,
                    max_seqlen_q=local_max_len,
                    max_seqlen_k=local_max_len,
                    window_size=(window_size, 0),
                    softmax_scale=softmax_scale,
                    causal=True,
                    global_attn_out=attn_out,
                    global_softmax_lse=softmax_lse,
                )
            elif min(first_kv_len, window_size) > (i - 1) * seq_len:
                # get qkv len for full attention block
                valid_kv_len = min(first_kv_len - (i - 1) * seq_len, seq_len)
                valid_q_len = first_q_len
                valid_w_len = min(window_size - (i - 1) * seq_len, seq_len * 2)
                # cut valid len with sliding window
                if valid_w_len < valid_q_len:
                    valid_q_len = valid_w_len
                if valid_w_len < valid_kv_len:
                    valid_kv_len = valid_w_len
                # window size for flash attention
                window_size_left = valid_w_len - valid_q_len
                # print(f"[forward] rank: {cp_local_rank}, step: {i}, q_id: {q_id}, kv_id: {kv_id}, compute full block, window_size_left: {window_size_left}, valid_q_len: {valid_q_len}, valid_kv_len: {valid_kv_len}, valid_w_len: {valid_w_len}")
                # compute attention
                flash_attn_fwd_func(
                    q[:, :valid_q_len],
                    kv[:, :, -valid_kv_len:],
                    window_size=(window_size_left, seq_len),
                    softmax_scale=softmax_scale,
                    causal=False,
                    global_attn_out=attn_out[:, :valid_q_len],
                    global_softmax_lse=softmax_lse[:, :, :valid_q_len],
                )
            else:
                # print(f"[forward] rank: {cp_local_rank}, step: {i}, q_id: {q_id}, kv_id: {kv_id}, no compute needed")
                pass

        # cast dtype and squeeze batch dim
        attn_out = attn_out.to(v.dtype).squeeze(0)
        softmax_lse = softmax_lse.to(torch.float32).squeeze(0)
        q = q.squeeze(0)
        kv = kv.squeeze(1)

        ctx.save_for_backward(q, kv, attn_out, softmax_lse, local_cu_seqlens)
        ctx.local_max_len = local_max_len
        ctx.first_q_len = first_q_len
        ctx.first_kv_len = first_kv_len
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
        q, kv, o, softmax_lse, local_cu_seqlens = ctx.saved_tensors
        seq_len = q.shape[0]

        # unsqueeze q, kv to batch_size = 1
        q = q.unsqueeze(0)
        kv = kv.unsqueeze(1)
        softmax_lse = softmax_lse.unsqueeze(0)
        o = o.unsqueeze(0)
        do = do.unsqueeze(0)

        # get args
        local_max_len = ctx.local_max_len
        first_q_len = ctx.first_q_len
        first_kv_len = ctx.first_kv_len
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
                # print(f"[backward] rank: {cp_local_rank}, step: {i}, q_id: {q_id}, kv_id: {kv_id}, compute local block, local_cu_seqlens: {local_cu_seqlens.tolist()}, local_max_len: {local_max_len}")
                flash_attn_varlen_bwd_func(
                    q,
                    kv,
                    o,
                    softmax_lse,
                    do,
                    dq_local,
                    dkv_local,
                    cu_seqlens_q=local_cu_seqlens,
                    cu_seqlens_k=local_cu_seqlens,
                    max_seqlen_q=local_max_len,
                    max_seqlen_k=local_max_len,
                    window_size=(window_size, 0),
                    softmax_scale=softmax_scale,
                    causal=True,
                )
                has_compute = True
            elif min(first_kv_len, window_size) > (fwd_i - 1) * seq_len:
                # get qkv len for full attention block
                valid_kv_len = min(first_kv_len - (fwd_i - 1) * seq_len, seq_len)
                valid_q_len = first_q_len
                valid_w_len = min(window_size - (fwd_i - 1) * seq_len, seq_len * 2)
                # cut valid len with sliding window
                if valid_w_len < valid_q_len:
                    valid_q_len = valid_w_len
                if valid_w_len < valid_kv_len:
                    valid_kv_len = valid_w_len
                # window size for flash attention
                window_size_left = valid_w_len - valid_q_len
                # set local dq dkv buffer to zero
                if valid_q_len < seq_len:
                    dq_local.zero_()
                if valid_kv_len < seq_len:
                    dkv_local.zero_()
                # print(f"[backward] rank: {cp_local_rank}, step: {i}, q_id: {q_id}, kv_id: {kv_id}, compute full block, window_size_left: {window_size_left}, valid_q_len: {valid_q_len}, valid_kv_len: {valid_kv_len}, valid_w_len: {valid_w_len}")
                flash_attn_bwd_func(
                    q[:, :valid_q_len],
                    kv[:, :, -valid_kv_len:],
                    o[:, :valid_q_len],
                    softmax_lse[
                        :, :, :valid_q_len
                    ].clone(),  # FIXME: avoid bugs at backward, don't know why
                    do[:, :valid_q_len],
                    dq_local[:, :valid_q_len],
                    dkv_local[:, :, -valid_kv_len:],
                    window_size=(window_size_left, seq_len),
                    softmax_scale=softmax_scale,
                    causal=False,
                )
                has_compute = True
            else:
                # print(f"[backward] rank: {cp_local_rank}, step: {i}, q_id: {q_id}, kv_id: {kv_id}, no compute needed")
                has_compute = False

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
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


def ring_sliding_window_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    cu_seqlens: torch.Tensor,
    cp_group: Optional[ProcessGroup] = None,
    cp_size: Optional[int] = None,
    cp_local_rank: Optional[int] = None,
    cp_prev_global_rank: Optional[int] = None,
    cp_next_global_rank: Optional[int] = None,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Ring sliding window attention with variable length.

    Args:
        q (torch.Tensor): Query tensor at current rank, shape [seq_len, num_q_heads, head_dim].
        k (torch.Tensor): Key tensor at current rank, shape [seq_len, num_kv_heads, head_dim].
        v (torch.Tensor): Value tensor at current rank, shape [seq_len, num_kv_heads, head_dim].
        window_size (int): Window size, the number of tokens that each query will attend to, similar to window_size[0] in flash_attn.
        cu_seqlens (torch.Tensor): Cumulative sequence lengths of the whole batch across all ranks, shape [batch_size + 1].
        cp_group (ProcessGroup): Process group for context parallelism. If None, use single GPU version.
        cp_size (Optional[int]): Number of ranks in the process group. If None, use torch.distributed.get_world_size(cp_group).
        cp_local_rank (Optional[int]): Local rank in cp_group. If None, use torch.distributed.get_rank(group=cp_group).
        cp_prev_global_rank (Optional[int]): Previous global rank. If None, will automatically compute the previous global rank from cp_group.
        cp_next_global_rank (Optional[int]): Next global rank. If None, will automatically compute the next global rank from cp_group.
        softmax_scale (Optional[float], optional): Softmax scale. Defaults to None, which is head_dim ** (-0.5).

    Returns:
        torch.Tensor: Attention output at current rank, shape [seq_len, num_q_heads, head_dim]

    Note:
        If don't set cp_prev_global_rank, cp_next_global_rank, please make sure the data is split and placed in the correct rank.
        You can use `rank_map = ring_swa.ops.get_group_local_to_global_rank_map(cp_group)` to get the rank map for the process group.
        We assume that the i'th sequence chunk is placed in the global rank `rank_map[i]`.
    """
    # here cu_seqlens is the cumulative sequence lengths of all tokens, not only for current rank
    # so we need to broadcast cu_seqlens to all ranks before calling this function
    assert (
        q.shape[0] == cu_seqlens[-1] // cp_size
    ), "cu_seqlens does not match with sequence length per rank"
    if cp_group is None:
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            window_size=(window_size, 0),
            softmax_scale=softmax_scale,
            causal=True,
        )
    else:
        if cp_size is None:
            cp_size = torch.distributed.get_world_size(cp_group)
        if cp_local_rank is None:
            cp_local_rank = torch.distributed.get_rank(group=cp_group)
        if cp_prev_global_rank is None or cp_next_global_rank is None:
            rank_map = get_group_local_to_global_rank_map(cp_group)
            cp_prev_global_rank = rank_map[(cp_local_rank - 1 + cp_size) % cp_size]
            cp_next_global_rank = rank_map[(cp_local_rank + 1) % cp_size]
        return RingSlidingWindowAttnVarlen.apply(
            q,
            k,
            v,
            window_size,
            cu_seqlens,
            None,
            cp_group,
            cp_size,
            cp_local_rank,
            cp_prev_global_rank,
            cp_next_global_rank,
            None,
            softmax_scale,
        )

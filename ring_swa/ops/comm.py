import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, Work
from typing import List


def ring_attn_p2p_communicate(
    rank: int,
    send_tensor: torch.Tensor,
    send_rank: int,
    recv_tensor: torch.Tensor,
    recv_rank: int,
    cp_group: ProcessGroup,
) -> List[Work]:
    """Point-to-point communications of KV and dKV in Attention with context parallelism.
    In forward pass, send_rank is the next rank, and recv_rank is the previous rank.
    In backward pass, send_rank is the previous rank, and recv_rank is the next rank.

    Args:
        rank: local rank for this rank in cp_group, range from [0, cp_size)
        send_tensor: the tensor to send from this rank
        send_rank: the global rank in cp_group to send the send_tensor from this rank to
        recv_tensor: the tensor to receive to this rank
        recv_rank: the global rank in cp_group to receive the recv_tensor from to this rank
        cp_group: the process group for context parallelism
    """
    send_recv_reqs = []
    # avoid deadlock, if rank is even, send first, then recv, otherwise recv first, then send
    if rank % 2 == 0:
        send_req = dist.isend(send_tensor, send_rank, cp_group)
        recv_req = dist.irecv(recv_tensor, recv_rank, cp_group)
        send_recv_reqs.append(send_req)
        send_recv_reqs.append(recv_req)
    else:
        recv_req = dist.irecv(recv_tensor, recv_rank, cp_group)
        send_req = dist.isend(send_tensor, send_rank, cp_group)
        send_recv_reqs.append(recv_req)
        send_recv_reqs.append(send_req)
    return send_recv_reqs

import os
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, Work, ReduceOp
from typing import Dict, List, Tuple


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


def broadcast_tensor_to_group(
    tensor: torch.Tensor,
    src_rank: int,
    cp_group: ProcessGroup,
) -> Work:
    """Broadcasts a tensor from a source rank to all other ranks in a process group.

    This function must be called by all ranks in the process group.

    Args:
        tensor (torch.Tensor): On the source rank, the tensor to send. On all
            other ranks, the tensor to be populated with the received data.
            The tensor must have the same shape and dtype across all ranks.
        src_rank (int): The global rank in the cp_group that will send the tensor.
        cp_group (ProcessGroup): The process group for the communication.

    Returns:
        Work: A single asynchronous work handle for the
            broadcast operation. The caller can use .wait() on this handle
            to ensure the operation is complete.
    """
    # The dist.broadcast operation is a collective, meaning all ranks in the
    # group must call it. The `src` parameter determines which rank sends
    # the data, and all other ranks receive it into the provided tensor.
    # We use async_op=True to make it non-blocking, which returns a work handle.
    broadcast_work = dist.broadcast(tensor, src=src_rank, group=cp_group, async_op=True)
    return broadcast_work


def reduce_sum_tesnsor_to_rank(
    tensor: torch.Tensor,
    dst_rank: int,
    cp_group: ProcessGroup,
) -> torch.Tensor:
    """Reduces tensors from all ranks by summing them and sends the result to a destination rank.

    This function must be called by all ranks in the process group.

    Args:
        tensor (torch.Tensor): The tensor from the local rank to be included in the sum.
            The tensor must have the same shape and dtype across all ranks. On the
            destination rank, this tensor will be overwritten with the summed result.
        dst_rank (int): The global rank in the cp_group that will receive the final sum.
        cp_group (ProcessGroup): The process group for the communication.

    Returns:
        torch.Tensor: A single tensor that is the sum of all tensors received by this rank.
    """
    # The dist.reduce operation is a collective that combines tensors from all ranks.
    # The `op` parameter specifies the operation (in this case, SUM).
    # The `dst` parameter is the rank that will receive the final result.
    # We use async_op=True to make it non-blocking.
    reduce_work = dist.reduce(
        tensor, dst=dst_rank, op=ReduceOp.SUM, group=cp_group, async_op=True
    )
    reduce_work.wait()
    return tensor


def exchange_and_sum_tensors(
    tensor_to_send: torch.Tensor,
    send_to_rank: int,
    cp_group: ProcessGroup,
) -> torch.Tensor:
    """Performs a personalized exchange and returns the sum of received tensors.

    This is a synchronous/blocking function. Each rank sends its `tensor_to_send`
    to a specific `send_to_rank`. The function handles the communication, waits
    for it to complete, and returns a single tensor representing the sum of all
    tensors received by the current rank.

    This function must be called by all ranks in the process group.

    Args:
        tensor_to_send (torch.Tensor): The tensor from the local rank to be sent.
            Must have the same shape and dtype across all ranks.
        send_to_rank (int): The destination rank in the cp_group for this rank's tensor.
        cp_group (ProcessGroup): The process group for the communication.

    Returns:
        torch.Tensor: A single tensor that is the sum of all tensors received by this rank.
    """
    world_size = dist.get_world_size(group=cp_group)

    # 1. Prepare the list of tensors to send for the all-to-all operation.
    # We place our tensor in the slot corresponding to the destination rank
    # and zero-tensors in all other slots.
    input_list = [torch.zeros_like(tensor_to_send) for _ in range(world_size)]
    input_list[send_to_rank] = tensor_to_send

    # 2. Prepare the list that will receive the incoming tensors.
    output_list = [torch.zeros_like(tensor_to_send) for _ in range(world_size)]

    # 3. Perform the all-to-all communication.
    # This is an async operation that returns a handle.
    work = dist.all_to_all(output_list, input_list, group=cp_group, async_op=True)

    # 4. Wait for the communication to complete. This makes the function synchronous.
    work.wait()

    # 5. Sum the received tensors.
    # We stack the list of tensors into a new dimension and sum along it.
    # This correctly handles receiving multiple tensors (if multiple ranks send
    # to us) and receiving zero tensors (the sum of zeros is zero).
    summed_tensor = torch.sum(torch.stack(output_list), dim=0)

    return summed_tensor


class _RankMapManager:
    """
    A manager class to compute and cache mappings from a group-local rank
    to a global rank for any given process group.

    This class is intended for internal use. The public API is the
    `get_group_local_to_global_rank_map` function.
    """

    def __init__(self):
        # The cache is an instance variable, encapsulated within this class.
        self._cache: Dict[Tuple[int, ...], Dict[int, int]] = {}

    def get_map(self, group: dist.ProcessGroup) -> Dict[int, int]:
        """
        Retrieves the group-local to global rank map for the given group.
        """
        if group is None:
            return {}

        cache_key = tuple(dist.get_process_group_ranks(group))

        if cache_key in self._cache:
            return self._cache[cache_key]

        rank_map = self._compute_map(group)
        self._cache[cache_key] = rank_map
        return rank_map

    def _compute_map(self, group: dist.ProcessGroup) -> Dict[int, int]:
        """
        The core logic to compute the rank map using an all_gather collective.
        """
        global_rank = dist.get_rank()
        group_local_rank = dist.get_rank(group=group)

        node_local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cpu")
        if dist.get_backend() == "nccl":
            device = torch.device(f"cuda:{node_local_rank}")

        rank_pair_tensor = torch.tensor(
            [group_local_rank, global_rank], dtype=torch.long, device=device
        )

        group_size = dist.get_world_size(group=group)
        gathered_list = [torch.empty_like(rank_pair_tensor) for _ in range(group_size)]

        dist.all_gather(gathered_list, rank_pair_tensor, group=group)

        rank_map = {}
        for pair_tensor in gathered_list:
            g_l_rank, g_rank = pair_tensor.tolist()
            rank_map[g_l_rank] = g_rank

        return rank_map

    def clear_cache(self):
        """Clears the internal cache."""
        self._cache.clear()
        print("Rank map cache has been cleared.")


_manager = _RankMapManager()


def get_group_local_to_global_rank_map(group: dist.ProcessGroup) -> Dict[int, int]:
    """
    Retrieves the group-local to global rank map for the given group.

    This function provides a simple, functional interface and uses a managed,
    global cache to avoid re-computing the map for the same group. This method
    must be called by all processes in the target group.

    Args:
        group (dist.ProcessGroup): The process group to get the map for.

    Returns:
        A dictionary mapping each group-local rank to its corresponding global rank.
    """
    return _manager.get_map(group)

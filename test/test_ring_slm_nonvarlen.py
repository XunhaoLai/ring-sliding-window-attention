import math
import os
import torch
import torch.distributed as dist
from ring_swa.streaming_llm.naive import naive_streaming_llm_attn
from ring_swa.streaming_llm.nonvarlen import ring_streaming_llm_attn


def setup_distributed():
    """Initializes the distributed environment."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size < 2:
        raise RuntimeError(
            "This script requires a multi-GPU setup. "
            "Please run with 'torchrun --nproc_per_node=N' where N > 1."
        )
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())


def cleanup_distributed():
    """Cleans up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def global_warmup(config: dict, warmup_times: int = 1):
    """
    Performs a single, global warm-up run to compile CUDA kernels
    before any tests start.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_seq_len = config["seq_len"] // world_size
    num_q_heads, num_kv_heads, head_dim = (
        config["num_q_heads"],
        config["num_kv_heads"],
        config["head_dim"],
    )
    dtype, window_size = config["dtype"], config["window_size"]

    for _ in range(warmup_times):
        q_local = torch.randn(
            1,
            local_seq_len,
            num_q_heads,
            head_dim,
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        k_local = torch.randn(
            1,
            local_seq_len,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        v_local = torch.randn(
            1,
            local_seq_len,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        do_local = torch.randn(
            1, local_seq_len, num_q_heads, head_dim, dtype=dtype, device="cuda"
        )

        o_warmup_flash = naive_streaming_llm_attn(
            q_local, k_local, v_local, window_size=window_size
        )
        o_warmup_flash.backward(do_local)
        q_local.grad.zero_()
        k_local.grad.zero_()
        v_local.grad.zero_()

        o_warmup_ring = ring_streaming_llm_attn(
            q_local,
            k_local,
            v_local,
            window_size,
            cp_group=dist.group.WORLD,
            cp_size=world_size,
            cp_local_rank=rank,
            cp_prev_global_rank=(rank - 1 + world_size) % world_size,
            cp_next_global_rank=(rank + 1) % world_size,
        )
        o_warmup_ring.backward(do_local)

    dist.barrier()


def generate_data(
    seq_len, num_q_heads, num_kv_heads, head_dim, dtype, window_size, world_size
):
    """Generates reference data and chunks on rank 0."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    q = torch.randn(
        1,
        seq_len,
        num_q_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    k = torch.randn(
        1,
        seq_len,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    v = torch.randn(
        1,
        seq_len,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    do = torch.randn(1, seq_len, num_q_heads, head_dim, dtype=dtype, device="cuda")

    start_event.record()
    o = naive_streaming_llm_attn(q, k, v, window_size=window_size)
    end_event.record()
    torch.cuda.synchronize()  # Wait for the operation to complete
    flash_forward_ms = start_event.elapsed_time(end_event)

    start_event.record()
    o.backward(do)
    end_event.record()
    torch.cuda.synchronize()  # Wait for the operation to complete
    flash_backward_ms = start_event.elapsed_time(end_event)

    timing_info = {
        "flash_forward_ms": flash_forward_ms,
        "flash_backward_ms": flash_backward_ms,
    }
    grad_q, grad_k, grad_v = q.grad, k.grad, v.grad

    q_chunks = [t.contiguous() for t in q.clone().detach().chunk(world_size, dim=1)]
    k_chunks = [t.contiguous() for t in k.clone().detach().chunk(world_size, dim=1)]
    v_chunks = [t.contiguous() for t in v.clone().detach().chunk(world_size, dim=1)]
    o_chunks = [t.contiguous() for t in o.clone().detach().chunk(world_size, dim=1)]
    do_chunks = [t.contiguous() for t in do.clone().detach().chunk(world_size, dim=1)]
    grad_q_chunks = [
        t.contiguous() for t in grad_q.clone().detach().chunk(world_size, dim=1)
    ]
    grad_k_chunks = [
        t.contiguous() for t in grad_k.clone().detach().chunk(world_size, dim=1)
    ]
    grad_v_chunks = [
        t.contiguous() for t in grad_v.clone().detach().chunk(world_size, dim=1)
    ]

    if dist.get_rank() == 0:
        print("\n" + "=" * 110)
        print(
            f"Test Config: dtype={str(dtype).split('.')[-1]}, seq_len={seq_len}, window_size={window_size}"
        )
        print(f"Q Heads: {num_q_heads}, KV Heads: {num_kv_heads}, Head Dim: {head_dim}")
        print(f"World Size: {world_size}, seq_len per rank: {seq_len // world_size}")

    data_chunks = (
        q_chunks,
        k_chunks,
        v_chunks,
        o_chunks,
        do_chunks,
        grad_q_chunks,
        grad_k_chunks,
        grad_v_chunks,
    )
    return data_chunks, timing_info


def scatter_data(data_chunks, shape, dtype, src=0):
    data_local = torch.empty(*shape, dtype=dtype, device="cuda")
    dist.scatter(data_local, scatter_list=data_chunks, src=src)
    return data_local


def compute_error(x, x_ref):
    """Computes max, 99th, and 99.9th percentile error."""
    diff = (x - x_ref).abs()
    max_err = diff.max().item()

    total_elements = diff.numel()
    if total_elements == 0:
        return max_err, 0.0, 0.0

    # p99 (top 1%)
    k_p99 = max(1, math.ceil(0.01 * total_elements))
    top_k_p99_errors = torch.topk(diff.flatten(), k_p99).values
    p99_err = top_k_p99_errors.min().item()

    # p99.9 (top 0.1%)
    k_p999 = max(1, math.ceil(0.001 * total_elements))
    top_k_p999_errors = torch.topk(diff.flatten(), k_p999).values
    p999_err = top_k_p999_errors.min().item()

    return max_err, p99_err, p999_err


def check_correctness(a: torch.Tensor, b_ref: torch.Tensor, atol: float, rtol: float):
    """
    Checks if two tensors are close and returns pass/fail status and mismatch percentage.
    """
    tolerance_check = torch.le(torch.abs(a - b_ref), atol + rtol * torch.abs(b_ref))
    num_mismatched = torch.sum(~tolerance_check).item()
    total_elements = a.numel()
    mismatch_pct = (num_mismatched / total_elements) * 100
    if mismatch_pct < 0.01:
        return True, 0.0
    passed = num_mismatched == 0
    return passed, mismatch_pct


def test_swa_nonvarlen(config: dict):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    seq_len, num_q_heads, num_kv_heads, head_dim = (
        config["seq_len"],
        config["num_q_heads"],
        config["num_kv_heads"],
        config["head_dim"],
    )
    window_size, dtype, atol, rtol = (
        config["window_size"],
        config["dtype"],
        config["atol"],
        config["rtol"],
    )
    cp_group = dist.group.WORLD
    cp_size, cp_local_rank = world_size, rank
    cp_prev_global_rank = (rank - 1 + cp_size) % cp_size
    cp_next_global_rank = (rank + 1) % cp_size

    flash_timing = {}
    if rank == 0:
        (
            q_chunks,
            k_chunks,
            v_chunks,
            ref_o_chunks,
            do_chunks,
            ref_grad_q_chunks,
            ref_grad_k_chunks,
            ref_grad_v_chunks,
        ), flash_timing = generate_data(
            seq_len, num_q_heads, num_kv_heads, head_dim, dtype, window_size, world_size
        )
    else:
        q_chunks = k_chunks = v_chunks = ref_o_chunks = do_chunks = None
        ref_grad_q_chunks = ref_grad_k_chunks = ref_grad_v_chunks = None

    local_seq_len = seq_len // world_size
    q_local = scatter_data(q_chunks, (1, local_seq_len, num_q_heads, head_dim), dtype)
    k_local = scatter_data(k_chunks, (1, local_seq_len, num_kv_heads, head_dim), dtype)
    v_local = scatter_data(v_chunks, (1, local_seq_len, num_kv_heads, head_dim), dtype)
    ref_o_local = scatter_data(
        ref_o_chunks, (1, local_seq_len, num_q_heads, head_dim), dtype
    )
    do_local = scatter_data(do_chunks, (1, local_seq_len, num_q_heads, head_dim), dtype)
    ref_grad_q = scatter_data(
        ref_grad_q_chunks, (1, local_seq_len, num_q_heads, head_dim), dtype
    )
    ref_grad_k = scatter_data(
        ref_grad_k_chunks, (1, local_seq_len, num_kv_heads, head_dim), dtype
    )
    ref_grad_v = scatter_data(
        ref_grad_v_chunks, (1, local_seq_len, num_kv_heads, head_dim), dtype
    )

    q_local.requires_grad = True
    k_local.requires_grad = True
    v_local.requires_grad = True

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    dist.barrier()
    start_event.record()
    o_local = ring_streaming_llm_attn(
        q_local,
        k_local,
        v_local,
        window_size,
        cp_group,
        cp_size,
        cp_local_rank,
        cp_prev_global_rank,
        cp_next_global_rank,
    )
    end_event.record()
    torch.cuda.synchronize()  # Wait for the forward pass to complete
    forward_time_ms = start_event.elapsed_time(end_event)

    dist.barrier()
    start_event.record()
    o_local.backward(do_local)
    end_event.record()
    torch.cuda.synchronize()  # Wait for the backward pass to complete
    backward_time_ms = start_event.elapsed_time(end_event)

    results = {"forward_ms": forward_time_ms, "backward_ms": backward_time_ms}
    if rank == 0:
        results.update(flash_timing)

    results["o_max"], results["o_p99_err"], results["o_p999_err"] = compute_error(
        o_local, ref_o_local
    )
    (
        results["q_grad_max"],
        results["q_grad_p99_err"],
        results["q_grad_p999_err"],
    ) = compute_error(q_local.grad, ref_grad_q)
    (
        results["k_grad_max"],
        results["k_grad_p99_err"],
        results["k_grad_p999_err"],
    ) = compute_error(k_local.grad, ref_grad_k)
    (
        results["v_grad_max"],
        results["v_grad_p99_err"],
        results["v_grad_p999_err"],
    ) = compute_error(v_local.grad, ref_grad_v)

    results["o_passed"], results["o_mismatch_pct"] = check_correctness(
        o_local, ref_o_local, atol, rtol
    )
    results["q_grad_passed"], results["q_grad_mismatch_pct"] = check_correctness(
        q_local.grad, ref_grad_q, atol, rtol
    )
    results["k_grad_passed"], results["k_grad_mismatch_pct"] = check_correctness(
        k_local.grad, ref_grad_k, atol, rtol
    )
    results["v_grad_passed"], results["v_grad_mismatch_pct"] = check_correctness(
        v_local.grad, ref_grad_v, atol, rtol
    )

    all_results = [None] * world_size
    dist.gather_object(results, all_results if rank == 0 else None, dst=0)

    if rank == 0:
        print_summary(all_results)


def print_summary(results_list: list):
    ref_timing = results_list[0]
    fwd_ref = ref_timing.get("flash_forward_ms", "N/A")
    bwd_ref = ref_timing.get("flash_backward_ms", "N/A")
    print("-" * 110)
    print(
        f"📈 Reference (flash_attn) | Forward: {fwd_ref:.3f} ms | Backward: {bwd_ref:.3f} ms"
    )

    global_fwd = max(r["forward_ms"] for r in results_list)
    global_bwd = max(r["backward_ms"] for r in results_list)
    print(
        f"🚀 Global Time (ring_swa)   | Forward: {global_fwd:.3f} ms | Backward: {global_bwd:.3f} ms"
    )

    print("-" * 110)
    print(
        f"{'Rank':<5} | {'Metric':<10} | {'Status (Mismatch %)':<22}  | {'P99 Error':<20} | {'P99.9 Error':<20} | {'Max Error':<20}"
    )
    print("-" * 110)

    for rank, res in enumerate(results_list):
        o_status = (
            "✅ PASS" if res["o_passed"] else f"❌ FAIL ({res['o_mismatch_pct']:.2f}%)"
        )
        q_status = (
            "✅ PASS"
            if res["q_grad_passed"]
            else f"❌ FAIL ({res['q_grad_mismatch_pct']:.2f}%)"
        )
        k_status = (
            "✅ PASS"
            if res["k_grad_passed"]
            else f"❌ FAIL ({res['k_grad_mismatch_pct']:.2f}%)"
        )
        v_status = (
            "✅ PASS"
            if res["v_grad_passed"]
            else f"❌ FAIL ({res['v_grad_mismatch_pct']:.2f}%)"
        )

        # UPDATED: Using p99 and p99.9 error keys for reporting
        print(
            f"{rank:<5} | {'Output':<10} | {o_status:<22} | {res['o_p99_err']:<20.5e} | {res['o_p999_err']:<20.5e} | {res['o_max']:<20.5e}"
        )
        print(
            f"{'':<5} | {'Grad (Q)':<10} | {q_status:<22} | {res['q_grad_p99_err']:<20.5e} | {res['q_grad_p999_err']:<20.5e} | {res['q_grad_max']:<20.5e}"
        )
        print(
            f"{'':<5} | {'Grad (K)':<10} | {k_status:<22} | {res['k_grad_p99_err']:<20.5e} | {res['k_grad_p999_err']:<20.5e} | {res['k_grad_max']:<20.5e}"
        )
        print(
            f"{'':<5} | {'Grad (V)':<10} | {v_status:<22} | {res['v_grad_p99_err']:<20.5e} | {res['v_grad_p999_err']:<20.5e} | {res['v_grad_max']:<20.5e}"
        )
        if rank < len(results_list) - 1:
            print("." * 110)
    print("=" * 110)


if __name__ == "__main__":
    # torchrun --nproc_per_node=4 test/test_ring_slm_nonvarlen.py
    torch.manual_seed(42)
    setup_distributed()
    try:
        base_config = {
            "num_q_heads": 32,
            "num_kv_heads": 4,
            "head_dim": 128,
            "atol": 2.5e-2,
            "rtol": 2.5e-2,
            "dtype": torch.bfloat16,
        }
        seq_lens = [4096, 8192, 16384, 32768]
        window_sizes = [1000, 4000, 8000]
        test_configs = [
            {**base_config, "seq_len": seq_len, "window_size": window_size}
            for seq_len in seq_lens
            for window_size in window_sizes
        ]

        # warmup
        if dist.get_rank() == 0:
            print("🚀 Performing GPU warm-up...")
        for config in test_configs:
            global_warmup(config)
        if dist.get_rank() == 0:
            print("✅ Warm-up complete.")

        # test
        for config in test_configs:
            test_swa_nonvarlen(config)

    finally:
        cleanup_distributed()

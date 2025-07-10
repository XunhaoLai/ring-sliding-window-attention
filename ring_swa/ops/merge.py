import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from importlib.metadata import version
from packaging.version import parse as parse_version


try:
    triton_version = parse_version(version("triton"))
except Exception:
    # Fallback if the package is not found or version is not parsable
    triton_version = parse_version("0.0.0")


def torch_merge_attn_out(
    attn_out: torch.Tensor,
    softmax_lse: torch.Tensor,
    new_attn_out: torch.Tensor,
    new_softmax_lse: torch.Tensor,
):
    """
    Merges attention outputs using a numerically stable method.

    Args:
        attn_out: the global attention output, will be updated in place, shape: (batch_size, seq_len, num_heads, head_dim)
        softmax_lse: the global softmax log-sum-exp, will be updated in place, shape: (batch_size, num_heads, seq_len)
        new_attn_out: the new attention output to be merged, shape: (batch_size, seq_len, num_heads, head_dim)
        new_softmax_lse: the new softmax log-sum-exp to be merged, shape: (batch_size, num_heads, seq_len)
    """
    lse_dtype = softmax_lse.dtype
    new_softmax_lse = new_softmax_lse.to(lse_dtype)

    # Calculate log-scale factors stably using softplus
    lse_diff = new_softmax_lse - softmax_lse
    log_scale = -F.softplus(lse_diff)
    log_new_scale = -F.softplus(-lse_diff)

    # Convert log-scales to scales
    # Reshape for broadcasting: [B, H, S] -> [B, S, H, 1]
    scale = torch.exp(log_scale).movedim(1, 2).unsqueeze(-1)
    new_scale = torch.exp(log_new_scale).movedim(1, 2).unsqueeze(-1)

    # Perform the weighted sum of attention outputs
    attn_out.mul_(scale).add_(new_attn_out * new_scale)

    # Safely merge softmax LSE using the log-sum-exp trick
    max_lse = torch.maximum(softmax_lse, new_softmax_lse)
    merged_lse = max_lse + torch.log1p(torch.exp(-torch.abs(lse_diff)))
    softmax_lse.copy_(merged_lse)


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": BN}, num_warps=nw, num_stages=ns)
        for BN in [64, 128, 256]
        for nw in [2, 4]
        for ns in [2, 3, 4]
    ],
    key=["head_dim"],
    restore_value=["global_attn_out", "global_lse"],
)
@triton.jit
def _merge_attn_out_kernel(
    global_attn_out,
    global_lse,
    local_attn_out,
    local_lse,
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    stride_ga_b,
    stride_ga_s,
    stride_ga_h,
    stride_ga_d,
    stride_gl_b,
    stride_gl_h,
    stride_gl_s,
    stride_la_b,
    stride_la_s,
    stride_la_h,
    stride_la_d,
    stride_ll_b,
    stride_ll_h,
    stride_ll_s,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_bh, pid_n = tl.program_id(0), tl.program_id(1)
    pid_b, pid_h = pid_bh // num_heads, pid_bh % num_heads
    # block pointers
    go_ptrs = tl.make_block_ptr(
        base=global_attn_out + pid_b * stride_ga_b + pid_h * stride_ga_h,
        shape=(seq_len, head_dim),
        strides=(stride_ga_s, stride_ga_d),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
        order=(1, 0),
    )
    lo_ptrs = tl.make_block_ptr(
        base=local_attn_out + pid_b * stride_la_b + pid_h * stride_la_h,
        shape=(seq_len, head_dim),
        strides=(stride_la_s, stride_la_d),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
        order=(1, 0),
    )
    gl_ptrs = tl.make_block_ptr(
        base=global_lse + pid_b * stride_gl_b + pid_h * stride_gl_h,
        shape=(seq_len,),
        strides=(stride_gl_s,),
        offsets=(pid_n * BLOCK_SIZE_N,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )
    ll_ptrs = tl.make_block_ptr(
        base=local_lse + pid_b * stride_ll_b + pid_h * stride_ll_h,
        shape=(seq_len,),
        strides=(stride_ll_s,),
        offsets=(pid_n * BLOCK_SIZE_N,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )

    # load values
    g_out = tl.load(go_ptrs, boundary_check=(0, 1))
    l_out = tl.load(lo_ptrs, boundary_check=(0, 1))
    g_lse = tl.load(gl_ptrs, boundary_check=(0,)).to(tl.float32)
    l_lse = tl.load(ll_ptrs, boundary_check=(0,)).to(tl.float32)

    # merge attn out
    lse_diff = l_lse - g_lse
    g_scale = tl.sigmoid(-lse_diff)[:, None]
    l_scale = tl.sigmoid(lse_diff)[:, None]
    g_out = g_out * g_scale + l_out * l_scale
    tl.store(go_ptrs, g_out.to(go_ptrs.dtype.element_ty), boundary_check=(0, 1))

    # merge and save lse
    max_lse = tl.maximum(g_lse, l_lse)
    merged_lse = max_lse + tl.log(1 + tl.exp(-tl.abs(lse_diff)))
    tl.store(gl_ptrs, merged_lse.to(gl_ptrs.dtype.element_ty), boundary_check=(0,))


def triton_merge_attn_out(
    attn_out: torch.Tensor,
    softmax_lse: torch.Tensor,
    new_attn_out: torch.Tensor,
    new_softmax_lse: torch.Tensor,
):
    """
    Merges attention outputs using a numerically stable method.

    Args:
        attn_out: the global attention output, will be updated in place, shape: (batch_size, seq_len, num_heads, head_dim)
        softmax_lse: the global softmax log-sum-exp, will be updated in place, shape: (batch_size, num_heads, seq_len)
        new_attn_out: the new attention output to be merged, shape: (batch_size, seq_len, num_heads, head_dim)
        new_softmax_lse: the new softmax log-sum-exp to be merged, shape: (batch_size, num_heads, seq_len)
    """
    batch_size, seq_len, num_heads, head_dim = attn_out.shape

    def grid(META):
        return (batch_size * num_heads, triton.cdiv(seq_len, META["BLOCK_SIZE_N"]))

    _merge_attn_out_kernel[grid](
        attn_out,
        softmax_lse,
        new_attn_out,
        new_softmax_lse,
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        attn_out.stride(0),
        attn_out.stride(1),
        attn_out.stride(2),
        attn_out.stride(3),
        softmax_lse.stride(0),
        softmax_lse.stride(1),
        softmax_lse.stride(2),
        new_attn_out.stride(0),
        new_attn_out.stride(1),
        new_attn_out.stride(2),
        new_attn_out.stride(3),
        new_softmax_lse.stride(0),
        new_softmax_lse.stride(1),
        new_softmax_lse.stride(2),
    )


if triton_version >= parse_version("3.0.0"):
    print(
        f"[RingSWA] Triton version {triton_version}, using Triton merge_attn_out function."
    )
    merge_attn_out = triton_merge_attn_out
else:
    if triton_version == parse_version("0.0.0"):
        print("[RingSWA] Triton is not installed, using Torch merge_attn_out function.")
    else:
        print(
            f"[RingSWA] Triton version {triton_version}, using Torch merge_attn_out function."
        )
    merge_attn_out = torch_merge_attn_out


if __name__ == "__main__":
    BATCH_SIZE = 8
    SEQ_LEN = 8192
    NUM_HEADS = 16
    HEAD_DIM = 128
    DEVICE = "cuda"

    # --- Create Test Data ---
    # Use requires_grad=False as these are just data tensors
    attn_out_torch = torch.randn(
        (BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=torch.float32, device=DEVICE
    )
    softmax_lse_torch = torch.randn(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN), dtype=torch.float64, device=DEVICE
    )
    new_attn_out = torch.randn(
        (BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE
    )
    new_softmax_lse = torch.randn(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN), dtype=torch.float32, device=DEVICE
    )

    # Clone inputs for the Triton version to have a fair comparison
    attn_out_triton = attn_out_torch.clone()
    softmax_lse_triton = softmax_lse_torch.clone()

    print("--- Verifying Correctness ---")
    # Run PyTorch version
    torch_merge_attn_out(
        attn_out_torch, softmax_lse_torch, new_attn_out, new_softmax_lse
    )

    # Run Triton version
    triton_merge_attn_out(
        attn_out_triton, softmax_lse_triton, new_attn_out, new_softmax_lse
    )

    # Compare results
    torch.testing.assert_close(attn_out_torch, attn_out_triton, atol=1e-2, rtol=1e-2)
    print("Attention Output All-Close")
    torch.testing.assert_close(
        softmax_lse_torch, softmax_lse_triton, atol=1e-5, rtol=1e-5
    )
    print("Softmax LSE All-Close")
    print("✅ Correctness check passed!")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=[1024 * 2**i for i in range(8)],  # 1k to 128k
            line_arg="provider",
            line_vals=["torch", "triton"],
            line_names=["Torch", "Triton"],
            styles=[("blue", "-"), ("green", "--")],
            ylabel="ms",
            plot_name="Merge Attention Output Performance (ms)",
            args={
                "batch_size": 1,
                "num_heads": 32,
                "head_dim": 128,
            },
        )
    )
    def benchmark_forward(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        provider,
    ):
        attn_out = torch.randn(
            (batch_size, seq_len, num_heads, head_dim),
            device="cuda",
            dtype=torch.float32,
        )
        softmax_lse = (
            torch.rand(
                (batch_size, num_heads, seq_len), device="cuda", dtype=torch.float64
            )
            * 5
        )
        new_attn_out = torch.randn(
            (batch_size, seq_len, num_heads, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        new_softmax_lse = (
            torch.rand(
                (batch_size, num_heads, seq_len), device="cuda", dtype=torch.float32
            )
            * 5
        )
        quantiles = [0.5, 0.2, 0.8]

        if provider == "torch":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch_merge_attn_out(
                    attn_out, softmax_lse, new_attn_out, new_softmax_lse
                ),
                quantiles=quantiles,
            )
        elif provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: triton_merge_attn_out(
                    attn_out, softmax_lse, new_attn_out, new_softmax_lse
                ),
                quantiles=quantiles,
            )

        return ms, min_ms, max_ms

    benchmark_forward.run(show_plots=True, print_data=True)

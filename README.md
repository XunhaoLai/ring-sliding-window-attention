# Ring Sliding Window Attention

This repository provides a Ring Sliding Window Attention implementation for efficient long-sequence training with context parallelism.

For a complete implementation details, please refer to our [documentation](docs/implementation_details.md).


## Features

  - **Ring Sliding Window Attention**: A distributed attention mechanism for sliding window attention.
  - **Ring Streaming LLM Attention**: A distributed attention mechanism for sliding window attention with attention sink token.
  - **Variable Length Support**: Supports variable-length sequences, similar to `flash_attn_varlen_func`.

## Installation

### Install with pip

```bash
pip install git+https://github.com/XunhaoLai/ring-sliding-window-attention.git
```

### Dependencies

  - Python \>= 3.8
  - torch \>= 2.7.0
  - flash\_attn \>= 2.5.8
  - einops \>= 0.6.0
  - triton \>= 3.0.0 (Optional)

## Usage

In ring sliding window attention, the query, key, and value tensors are split into cp_size chunks. Each chunk `i` is then placed on its corresponding rank `i` within the context parallel group. For simplicity, we'll use a random tensor in this example.

### Non-Varlen Example

The input query, key, and value tensors should be chunked and scattered to each rank.

```python
import torch
import torch.distributed as dist
from ring_swa import ring_sliding_window_attn

# Initialize distributed training
dist.init_process_group(backend="nccl")
torch.cuda.set_device(dist.get_rank())
cp_group = dist.group.WORLD
world_size = dist.get_world_size()
rank = dist.get_rank()

# Prepare input tensors; here we use random tensors at each rank
batch_size, seq_len, num_heads, head_dim = 2, 8192, 32, 128
q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
window_size = 1000

# Compute ring sliding window attention
output = ring_sliding_window_attn(
    q, k, v,
    window_size=window_size,
    cp_group=cp_group,
    cp_size=world_size,
    cp_local_rank=rank,
    cp_prev_global_rank=(rank - 1 + world_size) % world_size,
    cp_next_global_rank=(rank + 1) % world_size,
)

# Print attention output at each rank
print(f"[rank] {rank}, attention output shape: {output.shape}, norm: {output.norm()}")

dist.destroy_process_group()
```

### Varlen Example

This is similar to the non-varlen version, but requires an extra input, `cu_seqlens`. Note that `cu_seqlens` is based on all sequences in a batch, not just those related to the current rank.

```python
import torch
import torch.distributed as dist
from ring_swa import ring_sliding_window_attn_varlen

# Initialize distributed training
dist.init_process_group(backend="nccl")
torch.cuda.set_device(dist.get_rank())
cp_group = dist.group.WORLD
world_size = dist.get_world_size()
rank = dist.get_rank()

# Prepare input tensors; here we use random tensors at each rank
seq_len, num_heads, head_dim = 8192, 32, 128
cu_seqlens = torch.LongTensor([0, 512, 4000, 8192]).cuda().to(torch.int32)
q = torch.randn(seq_len // world_size, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
k = torch.randn(seq_len // world_size, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
v = torch.randn(seq_len // world_size, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
window_size = 1000

# Compute ring sliding window attention
output = ring_sliding_window_attn_varlen(
    q, k, v,
    window_size=window_size,
    cu_seqlens=cu_seqlens,
    cp_group=cp_group,
    cp_size=world_size,
    cp_local_rank=rank,
    cp_prev_global_rank=(rank - 1 + world_size) % world_size,
    cp_next_global_rank=(rank + 1) % world_size,
)

# Print attention output at each rank
print(f"[rank] {rank}, attention output shape: {output.shape}, norm: {output.norm()}")

dist.destroy_process_group()
```

### Streaming LLM Attention

The usage of `ring_streaming_llm_attn` and `ring_streaming_llm_attn_varlen` is similar to the standard ring sliding window attention functions. 
The key difference is its use of an attention sink, a mechanism that forces every query in the sequence to attend to the first token.

❗️**Note:** These functions currently assume that the first token for each sequence has identical content (e.g., a BOS token). Please ensure this condition is met before use.

```python
from ring_swa import ring_streaming_llm_attn
from ring_swa import ring_streaming_llm_attn_varlen
```

## Testing

You can run the test scripts to verify the correctness:

```bash
torchrun --nproc_per_node=4 test/test_ring_swa_nonvarlen.py
torchrun --nproc_per_node=4 test/test_ring_swa_varlen.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For any questions or feedback, you can contact laixunhao@pku.edu.cn.

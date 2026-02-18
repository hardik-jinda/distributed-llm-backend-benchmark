"""
DeepSpeed + Gloo 3D Tensor Collectives Benchmarking Script
Benchmarks collective operations using DeepSpeed communication APIs with Gloo backend
Uses 3D PyTorch tensors (batch, seq_len, hidden_dim) with bfloat16
"""

import torch
import deepspeed
import deepspeed.comm as dist
import json
import time
import os
import sys
from pathlib import Path

# Hyperparameters
BACKEND = "gloo"
IMPLEMENTATION_NAME = "deepspeed_gloo"

RANK_COUNTS = [4, 8, 16]

BATCH_SIZES = [1, 8, 16, 32]
SEQ_LENGTHS = [1, 2048, 4096, 8192]
HIDDEN_DIMS = [2048, 4096]

OPERATIONS = [
    "allreduce",
    "allgather",
    "broadcast",
    "gather",
    "reduce"
]

WARMUP_ITERATIONS = 10
MEASUREMENT_ITERATIONS = 100
OUTPUT_DIR = "results_3d_tensors_deepspeed_gloo"

def init_deepspeed():
    deepspeed.init_distributed(dist_backend=BACKEND)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print(f"DeepSpeed initialized")
        print(f"Backend: {BACKEND}")
        print(f"Using DeepSpeed communication APIs: deepspeed.comm")
    
    return rank, world_size

def benchmark_allreduce(rank, world_size, data, iterations):
    """Benchmark DeepSpeed allreduce with SUM operation"""
    timings = []
    
    for _ in range(iterations):
        data_copy = data.clone()
        dist.barrier()
        start = time.perf_counter()
        dist.all_reduce(data_copy, op=dist.ReduceOp.SUM)
        end = time.perf_counter()
        timings.append(end - start)
    
    return timings

def benchmark_allgather(rank, world_size, data, iterations):
    """Benchmark DeepSpeed allgather"""
    timings = []
    output_tensors = [torch.zeros_like(data) for _ in range(world_size)]
    
    for _ in range(iterations):
        dist.barrier()
        start = time.perf_counter()
        dist.all_gather(output_tensors, data)
        end = time.perf_counter()
        timings.append(end - start)
    
    return timings

def benchmark_broadcast(rank, world_size, data, iterations):
    """Benchmark DeepSpeed broadcast from rank 0"""
    timings = []
    
    for _ in range(iterations):
        data_copy = data.clone()
        dist.barrier()
        start = time.perf_counter()
        dist.broadcast(data_copy, src=0)
        end = time.perf_counter()
        timings.append(end - start)
    
    return timings

def benchmark_gather(rank, world_size, data, iterations):
    """Benchmark DeepSpeed gather to rank 0"""
    timings = []
    
    if rank == 0:
        gather_list = [torch.zeros_like(data) for _ in range(world_size)]
    else:
        gather_list = None
    
    for _ in range(iterations):
        dist.barrier()
        start = time.perf_counter()
        dist.gather(data, gather_list=gather_list, dst=0)
        end = time.perf_counter()
        timings.append(end - start)
    
    return timings

def benchmark_reduce(rank, world_size, data, iterations):
    """Benchmark DeepSpeed reduce with SUM operation to rank 0"""
    timings = []
    
    for _ in range(iterations):
        data_copy = data.clone()
        dist.barrier()
        start = time.perf_counter()
        dist.reduce(data_copy, dst=0, op=dist.ReduceOp.SUM)
        end = time.perf_counter()
        timings.append(end - start)
    
    return timings

def run_benchmark():
    rank, world_size = init_deepspeed()
    
    if world_size not in RANK_COUNTS:
        if rank == 0:
            print(f"Error: Current world size ({world_size}) not in RANK_COUNTS {RANK_COUNTS}")
        sys.exit(1)
    
    if rank == 0:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        print(f"Starting 3D Tensor benchmarks for {IMPLEMENTATION_NAME}")
        print(f"Backend: {BACKEND}")
        print(f"World size: {world_size}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"DeepSpeed version: {deepspeed.__version__}")
        print(f"Operations: {OPERATIONS}")
        print(f"Tensor shapes: (batch, seq_len, hidden_dim)")
        print(f"Batch sizes: {BATCH_SIZES}")
        print(f"Sequence lengths: {SEQ_LENGTHS}")
        print(f"Hidden dimensions: {HIDDEN_DIMS}")
        print(f"Data type: bfloat16")
        print(f"Warmup iterations: {WARMUP_ITERATIONS}")
        print(f"Measurement iterations: {MEASUREMENT_ITERATIONS}")
        print("-" * 80)
    
    operation_map = {
        "allreduce": benchmark_allreduce,
        "allgather": benchmark_allgather,
        "broadcast": benchmark_broadcast,
        "gather": benchmark_gather,
        "reduce": benchmark_reduce
    }
    
    for operation in OPERATIONS:
        for batch in BATCH_SIZES:
            for seq_len in SEQ_LENGTHS:
                for hidden_dim in HIDDEN_DIMS:
                    
                    tensor_shape = (batch, seq_len, hidden_dim)
                    num_elements = batch * seq_len * hidden_dim
                    tensor_size_bytes = num_elements * 2  # bfloat16 = 2 bytes
                    tensor_size_mb = tensor_size_bytes / (1024 * 1024)
                    
                    if rank == 0:
                        print(f"Benchmarking {operation}: shape={tensor_shape}, "
                              f"elements={num_elements}, size={tensor_size_mb:.2f} MB")
                    
                    # Generate random 3D tensor in bfloat16
                    torch.manual_seed(42 + rank)
                    data = torch.randn(*tensor_shape, dtype=torch.bfloat16)
                    
                    bench_func = operation_map[operation]
                    
                    # Warmup
                    try:
                        _ = bench_func(rank, world_size, data.clone(), WARMUP_ITERATIONS)
                    except Exception as e:
                        if rank == 0:
                            print(f"  ERROR during warmup: {e}")
                            import traceback
                            traceback.print_exc()
                        continue
                    
                    # Actual measurement
                    try:
                        timings = bench_func(rank, world_size, data.clone(), MEASUREMENT_ITERATIONS)
                    except Exception as e:
                        if rank == 0:
                            print(f"  ERROR during measurement: {e}")
                            import traceback
                            traceback.print_exc()
                        continue
                    
                    # Gather all timings to rank 0
                    timings_tensor = torch.tensor(timings, dtype=torch.float64)
                    
                    if rank == 0:
                        all_timings_list = [torch.zeros_like(timings_tensor) for _ in range(world_size)]
                    else:
                        all_timings_list = None
                    
                    dist.gather(timings_tensor, gather_list=all_timings_list, dst=0)
                    
                    if rank == 0:
                        # Convert tensors to lists
                        all_timings = [t.cpu().numpy().tolist() for t in all_timings_list]
                        
                        result = {
                            "implementation": IMPLEMENTATION_NAME,
                            "backend": BACKEND,
                            "operation": operation,
                            "num_ranks": world_size,
                            "tensor_shape": {
                                "batch": batch,
                                "seq_len": seq_len,
                                "hidden_dim": hidden_dim
                            },
                            "num_elements": num_elements,
                            "tensor_size_bytes": tensor_size_bytes,
                            "tensor_size_mb": tensor_size_mb,
                            "dtype": "bfloat16",
                            "warmup_iterations": WARMUP_ITERATIONS,
                            "measurement_iterations": MEASUREMENT_ITERATIONS,
                            "timing_method": "time.perf_counter()",
                            "timings": all_timings
                        }
                        
                        filename = f"{IMPLEMENTATION_NAME}_{operation}_ranks{world_size}_b{batch}_s{seq_len}_h{hidden_dim}.json"
                        filepath = os.path.join(OUTPUT_DIR, filename)
                        
                        with open(filepath, 'w') as f:
                            json.dump(result, f, indent=2)
                        
                        print(f"  Saved: {filename}")
    
    if rank == 0:
        print("-" * 80)
        print("3D Tensor benchmarking complete!")
        print(f"Results saved in '{OUTPUT_DIR}/' directory")
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception as e:
        rank = dist.get_rank() if dist.is_initialized() else "?"
        print(f"\n✗ [Rank {rank}] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
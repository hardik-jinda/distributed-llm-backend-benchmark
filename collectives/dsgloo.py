"""
DeepSpeed + Gloo Collectives Benchmarking Script
Benchmarks various collective operations using DeepSpeed communication APIs with Gloo backend
"""

import torch
import deepspeed
import deepspeed.comm as dist
import json
import time
import os
import sys
from pathlib import Path


# Backend configuration
BACKEND = "gloo"
IMPLEMENTATION_NAME = "deepspeed_gloo"

# Number of ranks to test (will use current world size)
RANK_COUNTS = [2, 4, 8, 16, 32, 56]

# Data sizes in number of elements (fp16)
DATA_SIZES = {
    "1KB": 256,           # 256 * 2 bytes = 512 bytes ≈ 1KB
    "64KB": 16384,        # 16384 * 2 bytes = 32KB ≈ 64KB
    "1MB": 262144,        # 262144 * 2 bytes = 524KB ≈ 1MB
    "16MB": 4194304,      # 4194304 * 2 bytes = 8MB ≈ 16MB
}

# Collective operations to benchmark
OPERATIONS = [
    "allreduce",
    "allgather",
    "broadcast",
    "gather",
    "scatter",
    "reduce",
    "alltoall"
]

# Iteration counts
WARMUP_ITERATIONS = 10
MEASUREMENT_ITERATIONS = 100

# Output directory
OUTPUT_DIR = "results/dsgloo"

# ============================================================================
# INITIALIZATION
# ============================================================================

def init_deepspeed():
    """Initialize DeepSpeed with Gloo backend"""
    
    # Initialize DeepSpeed distributed backend
    deepspeed.init_distributed(dist_backend=BACKEND)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print(f"DeepSpeed initialized")
        # print(f"Backend: {dist.get_backend()}")
        print(f"Using DeepSpeed communication APIs: deepspeed.comm")
    
    return rank, world_size

# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

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
    
    # Prepare output tensor list
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

def benchmark_scatter(rank, world_size, data, iterations):
    """Benchmark DeepSpeed scatter from rank 0"""
    timings = []
    
    if rank == 0:
        scatter_list = [data.clone() for _ in range(world_size)]
    else:
        scatter_list = None
    
    output = torch.zeros_like(data)
    
    for _ in range(iterations):
        dist.barrier()
        start = time.perf_counter()
        dist.scatter(output, scatter_list=scatter_list, src=0)
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

def benchmark_alltoall(rank, world_size, data, iterations):
    """Benchmark DeepSpeed alltoall"""
    timings = []
    
    # Split data into chunks for each rank
    chunk_size = len(data) // world_size
    input_splits = [data[i*chunk_size:(i+1)*chunk_size].clone() for i in range(world_size)]
    output_splits = [torch.zeros(chunk_size, dtype=data.dtype) for _ in range(world_size)]
    
    for _ in range(iterations):
        dist.barrier()
        start = time.perf_counter()
        dist.all_to_all(output_splits, input_splits)
        end = time.perf_counter()
        timings.append(end - start)
    
    return timings
    
    return timings

def benchmark_sendrecv(rank, world_size, data, iterations):
    """Benchmark DeepSpeed point-to-point send/recv in ring pattern"""
    timings = []
    
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size
    
    recv_buf = torch.zeros_like(data)
    
    for _ in range(iterations):
        dist.barrier()
        start = time.perf_counter()
        
        # Ring pattern: send to next, receive from previous
        send_req = dist.isend(data, dst=next_rank)
        recv_req = dist.irecv(recv_buf, src=prev_rank)
        
        send_req.wait()
        recv_req.wait()
        
        end = time.perf_counter()
        timings.append(end - start)
    
    return timings

# ============================================================================
# MAIN BENCHMARKING LOGIC
# ============================================================================

def run_benchmark():
    """Main benchmark execution function"""
    
    # Initialize DeepSpeed
    rank, world_size = init_deepspeed()
    
    # Check if current size is in our test configurations
    if world_size not in RANK_COUNTS:
        if rank == 0:
            print(f"Error: Current world size ({world_size}) not in RANK_COUNTS {RANK_COUNTS}")
            print(f"Please run with world size from: {RANK_COUNTS}")
        sys.exit(1)
    
    # Create output directory
    if rank == 0:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        print(f"Starting benchmarks for {IMPLEMENTATION_NAME}")
        print(f"Backend: {BACKEND}")
        print(f"World size: {world_size}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"DeepSpeed version: {deepspeed.__version__}")
        print(f"Operations: {OPERATIONS}")
        print(f"Data sizes: {list(DATA_SIZES.keys())}")
        print(f"Warmup iterations: {WARMUP_ITERATIONS}")
        print(f"Measurement iterations: {MEASUREMENT_ITERATIONS}")
        print(f"Timing method: time.perf_counter()")
        print("-" * 80)
    
    # Map operation names to functions
    operation_map = {
        "allreduce": benchmark_allreduce,
        "allgather": benchmark_allgather,
        "broadcast": benchmark_broadcast,
        "gather": benchmark_gather,
        "scatter": benchmark_scatter,
        "reduce": benchmark_reduce,
        "alltoall": benchmark_alltoall,
        "sendrecv": benchmark_sendrecv
    }
    
    # Benchmark each operation with each data size
    for operation in OPERATIONS:
        for size_name, num_elements in DATA_SIZES.items():
            
            if rank == 0:
                print(f"Benchmarking {operation} with {size_name} ({num_elements} elements)...")
            
            # Generate random data as PyTorch tensor (fp16)
            torch.manual_seed(42 + rank)  # Reproducible but different per rank
            data = torch.randn(num_elements, dtype=torch.float16)
            
            # Get benchmark function
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
            
            # Rank 0 saves results
            if rank == 0:
                # Convert tensors to lists
                all_timings = [t.cpu().numpy().tolist() for t in all_timings_list]
                
                # Create result dictionary
                result = {
                    "implementation": IMPLEMENTATION_NAME,
                    "backend": BACKEND,
                    "operation": operation,
                    "num_ranks": world_size,
                    "data_size_name": size_name,
                    "num_elements": num_elements,
                    "dtype": "float16",
                    "warmup_iterations": WARMUP_ITERATIONS,
                    "measurement_iterations": MEASUREMENT_ITERATIONS,
                    "timing_method": "time.perf_counter()",
                    "timings": all_timings  # List of lists: [rank][iteration]
                }
                
                # Create filename
                filename = f"{IMPLEMENTATION_NAME}_{operation}_ranks{world_size}_{size_name}.json"
                filepath = os.path.join(OUTPUT_DIR, filename)
                
                # Save to JSON
                with open(filepath, 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f"  Saved: {filename}")
    
    if rank == 0:
        print("-" * 80)
        print("Benchmarking complete!")
        print(f"Results saved in '{OUTPUT_DIR}/' directory")
    
    # Cleanup
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
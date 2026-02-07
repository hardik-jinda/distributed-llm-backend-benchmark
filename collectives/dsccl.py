"""
DeepSpeed + OneCCL Collectives Benchmarking Script
Benchmarks various collective operations using DeepSpeed communication APIs with OneCCL backend
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
BACKEND = "ccl"
IMPLEMENTATION_NAME = "deepspeed_oneccl"

RANK_COUNTS = [2, 4, 8, 16, 32, 56]

DATA_SIZES = {
    "1KB": 256,           
    "64KB": 16384,        
    "1MB": 262144,        
    "16MB": 4194304      
}


# OPERATIONS = [
#     "allreduce",
#     "allgather",
#     "broadcast",
#     "gather",
#     "scatter",
#     "reduce",
#     "alltoall",
#     "sendrecv"
# ]

OPERATIONS=[
    "gather"
]
WARMUP_ITERATIONS = 10
MEASUREMENT_ITERATIONS = 100
OUTPUT_DIR = "results/dsccl"

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
    timings = []
    if rank == 0:
        scatter_list = [data.clone() for _ in range(world_size)]
    else:
        scatter_list = None
    
    output = torch.zeros_like(data)
    for _ in range(iterations):
        dist.barrier()
        start = time.perf_counter()
        dist.scatter(output, scatter_list, src=0)
        end = time.perf_counter()
        timings.append(end - start)
    return timings

def benchmark_reduce(rank, world_size, data, iterations):
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
    timings = []
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

def benchmark_sendrecv(rank, world_size, data, iterations):
    timings = []
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size
    recv_buf = torch.zeros_like(data)
    
    for _ in range(iterations):
        dist.barrier()
        start = time.perf_counter()
        send_req = dist.isend(data, dst=next_rank)
        recv_req = dist.irecv(recv_buf, src=prev_rank)
        send_req.wait()
        recv_req.wait()
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
    
    for operation in OPERATIONS:
        for size_name, num_elements in DATA_SIZES.items():
            if rank == 0:
                print(f"Benchmarking {operation} with {size_name} ({num_elements} elements)...")
            
            torch.manual_seed(42 + rank)
            data = torch.randn(num_elements, dtype=torch.float16)
            bench_func = operation_map[operation]
            
            try:
                _ = bench_func(rank, world_size, data.clone(), WARMUP_ITERATIONS)
            except Exception as e:
                if rank == 0:
                    print(f"  ERROR during warmup: {e}")
                    import traceback
                    traceback.print_exc()
                continue
            
            try:
                timings = bench_func(rank, world_size, data.clone(), MEASUREMENT_ITERATIONS)
            except Exception as e:
                if rank == 0:
                    print(f"  ERROR during measurement: {e}")
                    import traceback
                    traceback.print_exc()
                continue
            
            timings_tensor = torch.tensor(timings, dtype=torch.float64)
            
            if rank == 0:
                all_timings_list = [torch.zeros_like(timings_tensor) for _ in range(world_size)]
            else:
                all_timings_list = None
            
            dist.gather(timings_tensor, gather_list=all_timings_list, dst=0)
            
            if rank == 0:
                all_timings = [t.cpu().numpy().tolist() for t in all_timings_list]
                
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
                    "timings": all_timings
                }
                
                filename = f"{IMPLEMENTATION_NAME}_{operation}_ranks{world_size}_{size_name}.json"
                filepath = os.path.join(OUTPUT_DIR, filename)
                
                with open(filepath, 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f"  Saved: {filename}")
    
    if rank == 0:
        print("-" * 80)
        print("Benchmarking complete!")
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
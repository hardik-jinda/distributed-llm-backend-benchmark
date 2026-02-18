"""
MPI 3D Tensor Collectives Benchmarking Script
Benchmarks MPI collective operations with 3D PyTorch tensors (batch, seq_len, hidden_dim)
Mimics real LLM communication patterns using bfloat16
"""

import torch
from mpi4py import MPI
import json
import time
import os
import sys
from pathlib import Path
import numpy as np

# Hyperparameters
MPI_IMPLEMENTATION = "openmpi"  # Change to "intelmpi" when testing Intel MPI

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
OUTPUT_DIR = "results_3d_tensors"

def benchmark_allreduce(comm, data, iterations):
    """Benchmark MPI_Allreduce with SUM operation"""
    rank = comm.Get_rank()
    timings = []
    
    # Convert bfloat16 to float32 for MPI compatibility
    data_fp32 = data.float().cpu().numpy()
    
    for _ in range(iterations):
        comm.Barrier()
        start = time.perf_counter()
        result = comm.allreduce(data_fp32, op=MPI.SUM)
        end = time.perf_counter()
        timings.append(end - start)
    
    return timings

def benchmark_allgather(comm, data, iterations):
    """Benchmark MPI_Allgather"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    timings = []
    
    data_fp32 = data.float().cpu().numpy()
    
    for _ in range(iterations):
        comm.Barrier()
        start = time.perf_counter()
        result = comm.allgather(data_fp32)
        end = time.perf_counter()
        timings.append(end - start)
    
    return timings

def benchmark_broadcast(comm, data, iterations):
    """Benchmark MPI_Broadcast from root=0"""
    rank = comm.Get_rank()
    timings = []
    
    data_fp32 = data.float().cpu().numpy()
    
    if rank == 0:
        send_data = data_fp32.copy()
    else:
        send_data = np.empty_like(data_fp32)
    
    for _ in range(iterations):
        comm.Barrier()
        start = time.perf_counter()
        comm.Bcast(send_data, root=0)
        end = time.perf_counter()
        timings.append(end - start)
    
    return timings

def benchmark_gather(comm, data, iterations):
    """Benchmark MPI_Gather to root=0"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    timings = []
    
    data_fp32 = data.float().cpu().numpy()
    
    for _ in range(iterations):
        comm.Barrier()
        start = time.perf_counter()
        result = comm.gather(data_fp32, root=0)
        end = time.perf_counter()
        timings.append(end - start)
    
    return timings

def benchmark_reduce(comm, data, iterations):
    """Benchmark MPI_Reduce with SUM operation to root=0"""
    rank = comm.Get_rank()
    timings = []
    
    data_fp32 = data.float().cpu().numpy()
    
    for _ in range(iterations):
        comm.Barrier()
        start = time.perf_counter()
        result = comm.reduce(data_fp32, op=MPI.SUM, root=0)
        end = time.perf_counter()
        timings.append(end - start)
    
    return timings

def run_benchmark():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size not in RANK_COUNTS:
        if rank == 0:
            print(f"Error: Current world size ({size}) not in RANK_COUNTS {RANK_COUNTS}")
            print(f"Please run with mpirun -n <ranks> where <ranks> is one of {RANK_COUNTS}")
        return
    
    if rank == 0:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        print(f"Starting 3D Tensor benchmarks for {MPI_IMPLEMENTATION}")
        print(f"World size: {size}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Operations: {OPERATIONS}")
        print(f"Tensor shapes: (batch, seq_len, hidden_dim)")
        print(f"Batch sizes: {BATCH_SIZES}")
        print(f"Sequence lengths: {SEQ_LENGTHS}")
        print(f"Hidden dimensions: {HIDDEN_DIMS}")
        print(f"Data type: bfloat16 (converted to float32 for MPI)")
        print(f"Note: Tensor size reported as bfloat16 (2 bytes/element)")
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
                    # Report size as if bfloat16 (2 bytes) even though we convert to fp32 for MPI
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
                        _ = bench_func(comm, data.clone(), WARMUP_ITERATIONS)
                    except Exception as e:
                        if rank == 0:
                            print(f"  ERROR during warmup: {e}")
                            import traceback
                            traceback.print_exc()
                        continue
                    
                    # Actual measurement
                    try:
                        timings = bench_func(comm, data.clone(), MEASUREMENT_ITERATIONS)
                    except Exception as e:
                        if rank == 0:
                            print(f"  ERROR during measurement: {e}")
                            import traceback
                            traceback.print_exc()
                        continue
                    
                    # Gather all timings to rank 0
                    all_timings = comm.gather(timings, root=0)
                    
                    if rank == 0:
                        result = {
                            "mpi_implementation": MPI_IMPLEMENTATION,
                            "operation": operation,
                            "num_ranks": size,
                            "tensor_shape": {
                                "batch": batch,
                                "seq_len": seq_len,
                                "hidden_dim": hidden_dim
                            },
                            "num_elements": num_elements,
                            "tensor_size_bytes": tensor_size_bytes,
                            "tensor_size_mb": tensor_size_mb,
                            "dtype": "bfloat16",
                            "dtype_note": "Generated as bf16, converted to fp32 for MPI, size reported as bf16",
                            "warmup_iterations": WARMUP_ITERATIONS,
                            "measurement_iterations": MEASUREMENT_ITERATIONS,
                            "timing_method": "time.perf_counter()",
                            "timings": all_timings
                        }
                        
                        filename = f"{MPI_IMPLEMENTATION}_{operation}_ranks{size}_b{batch}_s{seq_len}_h{hidden_dim}.json"
                        filepath = os.path.join(OUTPUT_DIR, filename)
                        
                        with open(filepath, 'w') as f:
                            json.dump(result, f, indent=2)
                        
                        print(f"  Saved: {filename}")
    
    if rank == 0:
        print("-" * 80)
        print("3D Tensor benchmarking complete!")
        print(f"Results saved in '{OUTPUT_DIR}/' directory")

if __name__ == "__main__":
    run_benchmark()
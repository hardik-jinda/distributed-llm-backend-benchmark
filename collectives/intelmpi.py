"""
MPI Collectives Benchmarking Script
Benchmarks various MPI collective operations with mpi4py
"""

import numpy as np
from mpi4py import MPI
import json
import time
import os
from pathlib import Path

# MPI Implementation (set this manually before running)
MPI_IMPLEMENTATION = "intelmpi"  # Change to "intelmpi" when testing Intel MPI

# Data type
DTYPE = np.float16

# Number of ranks to test (will use current world size)
RANK_COUNTS = [2, 4, 8, 16, 32, 56]

# Data sizes in number of elements
DATA_SIZES = {
    "1KB": 256,           # 256 * 2 bytes = 512 bytes ≈ 1KB
    "64KB": 16384,        # 16384 * 2 bytes = 32KB ≈ 64KB
    "1MB": 262144,        # 262144 * 2 bytes = 524KB ≈ 1MB
    "16MB": 4194304      # 4194304 * 2 bytes = 8MB ≈ 16MB
       # 67108864 * 2 bytes = 134MB ≈ 256MB
}



# Collective operations to benchmark
OPERATIONS = [
    "allreduce",
    "allgather",
    "broadcast",
    "gather",
    "scatter",
    "reduce",
    "alltoall",
    "sendrecv"
]

# Iteration counts
WARMUP_ITERATIONS = 10
MEASUREMENT_ITERATIONS = 100

# Output directory
OUTPUT_DIR = "results/intelmpi"

# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def benchmark_allreduce(comm, data, iterations):
    """Benchmark MPI_Allreduce with SUM operation"""
    rank = comm.Get_rank()
    timings = []
    
    for _ in range(iterations):
        comm.Barrier()
        start = MPI.Wtime()
        result = comm.allreduce(data, op=MPI.SUM)
        end = MPI.Wtime()
        timings.append(end - start)
    
    return timings

def benchmark_allgather(comm, data, iterations):
    """Benchmark MPI_Allgather"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    timings = []
    
    for _ in range(iterations):
        comm.Barrier()
        start = MPI.Wtime()
        result = comm.allgather(data)
        end = MPI.Wtime()
        timings.append(end - start)
    
    return timings

def benchmark_broadcast(comm, data, iterations):
    """Benchmark MPI_Broadcast from root=0"""
    rank = comm.Get_rank()
    timings = []
    
    # Only root has actual data, others have buffer
    if rank == 0:
        send_data = data.copy()
    else:
        send_data = np.empty_like(data)
    
    for _ in range(iterations):
        comm.Barrier()
        start = MPI.Wtime()
        comm.Bcast(send_data, root=0)
        end = MPI.Wtime()
        timings.append(end - start)
    
    return timings

def benchmark_gather(comm, data, iterations):
    """Benchmark MPI_Gather to root=0"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    timings = []
    
    for _ in range(iterations):
        comm.Barrier()
        start = MPI.Wtime()
        result = comm.gather(data, root=0)
        end = MPI.Wtime()
        timings.append(end - start)
    
    return timings

def benchmark_scatter(comm, data, iterations):
    """Benchmark MPI_Scatter from root=0"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    timings = []
    
    # Root needs to prepare data for all ranks
    if rank == 0:
        send_data = [data.copy() for _ in range(size)]
    else:
        send_data = None
    
    for _ in range(iterations):
        comm.Barrier()
        start = MPI.Wtime()
        result = comm.scatter(send_data, root=0)
        end = MPI.Wtime()
        timings.append(end - start)
    
    return timings

def benchmark_reduce(comm, data, iterations):
    """Benchmark MPI_Reduce with SUM operation to root=0"""
    rank = comm.Get_rank()
    timings = []
    
    for _ in range(iterations):
        comm.Barrier()
        start = MPI.Wtime()
        result = comm.reduce(data, op=MPI.SUM, root=0)
        end = MPI.Wtime()
        timings.append(end - start)
    
    return timings

def benchmark_alltoall(comm, data, iterations):
    """Benchmark MPI_Alltoall"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    timings = []
    
    # Split data into chunks for each rank
    chunk_size = len(data) // size
    send_data = [data[i*chunk_size:(i+1)*chunk_size].copy() for i in range(size)]
    
    for _ in range(iterations):
        comm.Barrier()
        start = MPI.Wtime()
        result = comm.alltoall(send_data)
        end = MPI.Wtime()
        timings.append(end - start)
    
    return timings

def benchmark_sendrecv(comm, data, iterations):
    """Benchmark point-to-point Send/Recv in ring pattern"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    timings = []
    
    next_rank = (rank + 1) % size
    prev_rank = (rank - 1 + size) % size
    
    recv_buf = np.empty_like(data)
    
    for _ in range(iterations):
        comm.Barrier()
        start = MPI.Wtime()
        
        # Ring pattern: send to next, receive from previous
        req_send = comm.Isend(data, dest=next_rank, tag=0)
        req_recv = comm.Irecv(recv_buf, source=prev_rank, tag=0)
        
        req_send.Wait()
        req_recv.Wait()
        
        end = MPI.Wtime()
        timings.append(end - start)
    
    return timings

# ============================================================================
# MAIN BENCHMARKING LOGIC
# ============================================================================

def run_benchmark():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Check if current size is in our test configurations
    if size not in RANK_COUNTS:
        if rank == 0:
            print(f"Error: Current world size ({size}) not in RANK_COUNTS {RANK_COUNTS}")
            print(f"Please run with mpirun -n <ranks> where <ranks> is one of {RANK_COUNTS}")
        return
    
    # Create output directory
    if rank == 0:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        print(f"Starting benchmarks for {MPI_IMPLEMENTATION}")
        print(f"World size: {size}")
        print(f"Operations: {OPERATIONS}")
        print(f"Data sizes: {list(DATA_SIZES.keys())}")
        print(f"Warmup iterations: {WARMUP_ITERATIONS}")
        print(f"Measurement iterations: {MEASUREMENT_ITERATIONS}")
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
            
            # Generate random data
            np.random.seed(42 + rank)  # Reproducible but different per rank
            data = np.random.randn(num_elements).astype(DTYPE)
            
            # Get benchmark function
            bench_func = operation_map[operation]
            
            # Warmup
            try:
                _ = bench_func(comm, data, WARMUP_ITERATIONS)
            except Exception as e:
                if rank == 0:
                    print(f"  ERROR during warmup: {e}")
                continue
            
            # Actual measurement
            try:
                timings = bench_func(comm, data, MEASUREMENT_ITERATIONS)
            except Exception as e:
                if rank == 0:
                    print(f"  ERROR during measurement: {e}")
                continue
            
            # Gather all timings to rank 0
            all_timings = comm.gather(timings, root=0)
            
            # Rank 0 saves results
            if rank == 0:
                # Create result dictionary
                result = {
                    "mpi_implementation": MPI_IMPLEMENTATION,
                    "operation": operation,
                    "num_ranks": size,
                    "data_size_name": size_name,
                    "num_elements": num_elements,
                    "dtype": str(DTYPE),
                    "warmup_iterations": WARMUP_ITERATIONS,
                    "measurement_iterations": MEASUREMENT_ITERATIONS,
                    "timings": all_timings  # List of lists: [rank][iteration]
                }
                
                # Create filename
                filename = f"{MPI_IMPLEMENTATION}_{operation}_ranks{size}_{size_name}.json"
                filepath = os.path.join(OUTPUT_DIR, filename)
                
                # Save to JSON
                with open(filepath, 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f"  Saved: {filename}")
    
    if rank == 0:
        print("-" * 80)
        print("Benchmarking complete!")
        print(f"Results saved in '{OUTPUT_DIR}/' directory")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_benchmark()
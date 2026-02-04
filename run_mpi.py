"""
MPI Backend Runner

Runs experiments using MPI-based backends (OpenMPI or IntelMPI).
Uses pure mpi4py for all MPI communication (no torch.distributed).
"""

import argparse
import os
import sys
import time
import torch
from mpi4py import MPI

from models import create_model_from_config
from data_gen import create_dataset_from_config
from utils import (
    load_config,
    MetricsCollector,
    Timer,
    setup_environment,
    run_experiment,
    print_summary,
    save_results,
    collect_system_info
)


def initialize_mpi_backend():
    """
    Initialize MPI using mpi4py only.
    
    Returns:
        rank: Process rank
        world_size: Total number of processes
        comm: MPI communicator
    """
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    return rank, world_size, comm


def cleanup_mpi_backend():
    """Cleanup MPI resources (barrier before exit)."""
    comm = MPI.COMM_WORLD
    comm.Barrier()


def main():
    parser = argparse.ArgumentParser(description='Run MPI backend experiment')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--backend', type=str, required=True,
                       choices=['openmpi', 'intelmpi'],
                       help='MPI backend name (for output file naming)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup environment variables
    setup_environment(config)
    
    # Initialize MPI and distributed backend
    init_start_time = time.time()
    rank, world_size, comm = initialize_mpi_backend()
    init_elapsed = time.time() - init_start_time
    
    # Verify world size matches config
    expected_world_size = config['parallelism']['world_size']
    if world_size != expected_world_size:
        if rank == 0:
            print(f"ERROR: World size mismatch. Expected {expected_world_size}, got {world_size}")
        sys.exit(1)
    
    # Print initialization info (rank 0 only)
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Experiment: {config['experiment']['name']}")
        print(f"Backend: {args.backend.upper()}")
        print(f"{'='*60}")
        print(f"World size: {world_size}")
        print(f"Initialization time: {init_elapsed:.4f}s")
        
        # Print system info
        system_info = collect_system_info()
        print(f"\nSystem Info:")
        print(f"  Platform: {system_info['platform']}")
        print(f"  Python: {system_info['python_version']}")
        print(f"  PyTorch: {system_info['torch_version']}")
        print(f"  CPUs (physical): {system_info['cpu_count']}")
        print(f"  CPUs (logical): {system_info['cpu_count_logical']}")
        print(f"  Total Memory: {system_info['total_memory_gb']:.2f} GB")
        
        # Print config
        print(f"\nModel: {config['model']['size']}")
        print(f"  Hidden size: {config['model']['hidden_size']}")
        print(f"  Layers: {config['model']['num_layers']}")
        print(f"  Heads: {config['model']['num_heads']}")
        
        print(f"\nInput:")
        print(f"  Batch size: {config['input']['batch_size']}")
        print(f"  Sequence length: {config['input']['sequence_length']}")
        
        print(f"\nExecution:")
        print(f"  Warmup iterations: {config['execution']['warmup_iterations']}")
        print(f"  Benchmark iterations: {config['execution']['benchmark_iterations']}")
        print(f"{'='*60}\n")
    
    # Barrier to ensure all ranks see the print
    comm.Barrier()
    
    # Create model
    if rank == 0:
        print(f"[Rank {rank}] Creating model...")
    
    model = create_model_from_config(config, rank, world_size, comm)
    
    if rank == 0:
        total_params = model.get_num_parameters()
        memory_per_rank = model.get_memory_footprint()
        print(f"[Rank {rank}] Model created:")
        print(f"  Total parameters: {total_params/1e9:.2f}B")
        print(f"  Memory per rank: {memory_per_rank/1e9:.2f} GB")
    
    comm.Barrier()
    
    # Create dataset
    if rank == 0:
        print(f"[Rank {rank}] Creating dataset...")
    
    dataset = create_dataset_from_config(config)
    
    if rank == 0:
        batch = dataset.get_batch()
        print(f"[Rank {rank}] Dataset created:")
        print(f"  Batch shape: {batch.shape}")
        print(f"  Batch dtype: {batch.dtype}")
        print(f"  Batch size: {batch.numel() * batch.element_size() / 1e6:.2f} MB")
    
    comm.Barrier()
    
    # Create metrics collector
    metrics = MetricsCollector(rank, world_size)
    metrics.record_init_time(init_elapsed)
    
    # Warmup phase
    if rank == 0:
        print(f"\n[Rank {rank}] Starting warmup: {config['execution']['warmup_iterations']} iterations")
    
    warmup_iters = config['execution']['warmup_iterations']
    for i in range(warmup_iters):
        batch = dataset.get_batch()
        
        with Timer() as t:
            output = model(batch)
        
        metrics.record_warmup_time(t.elapsed)
        
        if rank == 0 and (i + 1) % 10 == 0:
            print(f"  Warmup iteration {i+1}/{warmup_iters}")
    
    comm.Barrier()
    
    # Benchmark phase
    if rank == 0:
        print(f"\n[Rank {rank}] Starting benchmark: {config['execution']['benchmark_iterations']} iterations")
    
    benchmark_iters = config['execution']['benchmark_iterations']
    for i in range(benchmark_iters):
        batch = dataset.get_batch()
        
        # Synchronize before timing
        comm.Barrier()
        
        with Timer() as t:
            output = model(batch)
        
        # Synchronize after forward pass to get accurate timing
        comm.Barrier()
        
        metrics.record_forward_time(t.elapsed)
        
        if rank == 0 and (i + 1) % 20 == 0:
            print(f"  Benchmark iteration {i+1}/{benchmark_iters}")
    
    comm.Barrier()
    
    # Get local summary
    local_summary = metrics.get_summary()
    
    if rank == 0:
        print(f"\n[Rank {rank}] Benchmark complete. Processing results...")
    
    # Gather metrics from all ranks using mpi4py
    all_forward_means = comm.gather(local_summary['forward_mean'], root=0)
    
    rank_stats = None
    if rank == 0:
        import numpy as np
        means_array = np.array(all_forward_means)
        variance_across_ranks = float(np.var(means_array))
        cv = float(np.std(means_array) / np.mean(means_array))
        
        rank_stats = {
            'forward_mean_per_rank': means_array.tolist(),
            'variance_across_ranks': variance_across_ranks,
            'coefficient_of_variation': cv,
        }
    
    # Save results (rank 0 only)
    if rank == 0:
        # Combine all metrics
        results = {
            'experiment': config['experiment']['name'],
            'backend': args.backend,
            'config': config,
            'system_info': collect_system_info(),
            'rank_0_summary': local_summary,
            'rank_statistics': rank_stats,
            'raw_metrics_rank_0': metrics.get_raw_metrics(),
        }
        
        # Print summary
        print_summary(local_summary, args.backend.upper(), rank)
        
        if rank_stats:
            print(f"\nRank Statistics:")
            print(f"  Variance across ranks: {rank_stats['variance_across_ranks']:.6f}")
            print(f"  Coefficient of variation: {rank_stats['coefficient_of_variation']:.4f}")
            print(f"  Forward mean per rank: {rank_stats['forward_mean_per_rank']}")
        
        # Save to file
        output_dir = config['experiment']['output_dir']
        output_file = f"{output_dir}/{args.backend}_{config['experiment']['name']}.json"
        save_results(results, output_file, rank)
    
    # Cleanup
    comm.Barrier()
    cleanup_mpi_backend()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Experiment completed successfully!")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

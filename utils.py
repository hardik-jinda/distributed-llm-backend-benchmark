"""
Utility Functions

Metrics collection, timing, file I/O, system info, and common experiment logic.
"""

import time
import json
import yaml
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class MetricsCollector:
    """
    Collect and store experiment metrics.
    """
    
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.metrics = {
            'init_time': None,
            'warmup_times': [],
            'forward_times': [],
        }
    
    def record_init_time(self, elapsed):
        """Record backend initialization time."""
        self.metrics['init_time'] = elapsed
    
    def record_warmup_time(self, elapsed):
        """Record warmup iteration time."""
        self.metrics['warmup_times'].append(elapsed)
    
    def record_forward_time(self, elapsed):
        """Record benchmark iteration time."""
        self.metrics['forward_times'].append(elapsed)
    
    def get_summary(self):
        """
        Calculate summary statistics from collected metrics.
        
        Returns:
            Dictionary with mean, std, percentiles, etc.
        """
        forward_times = np.array(self.metrics['forward_times'])
        
        summary = {
            'rank': self.rank,
            'world_size': self.world_size,
            'init_time': self.metrics['init_time'],
            'num_iterations': len(forward_times),
            'forward_mean': float(np.mean(forward_times)),
            'forward_std': float(np.std(forward_times)),
            'forward_min': float(np.min(forward_times)),
            'forward_max': float(np.max(forward_times)),
            'forward_median': float(np.median(forward_times)),
            'forward_p95': float(np.percentile(forward_times, 95)),
            'forward_p99': float(np.percentile(forward_times, 99)),
        }
        
        return summary
    
    def get_raw_metrics(self):
        """Return raw metrics dictionary."""
        return self.metrics


class Timer:
    """
    Context manager for timing code blocks.
    """
    
    def __init__(self):
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_json(data, filepath):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def collect_system_info():
    """
    Collect system information.
    
    Returns:
        Dictionary with system details
    """
    import platform
    import psutil
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cpu_count': psutil.cpu_count(logical=False),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'total_memory_gb': psutil.virtual_memory().total / (1024**3),
    }
    
    return info


def setup_environment(config):
    """
    Setup environment variables from config.
    
    Args:
        config: Configuration dictionary
    """
    import os
    
    system_config = config['system']
    
    if 'omp_num_threads' in system_config:
        os.environ['OMP_NUM_THREADS'] = str(system_config['omp_num_threads'])
    
    if 'mkl_num_threads' in system_config:
        os.environ['MKL_NUM_THREADS'] = str(system_config['mkl_num_threads'])


def gather_metrics_from_all_ranks(local_summary, rank, world_size):
    """
    Gather metrics from all ranks to rank 0.
    
    Args:
        local_summary: Summary dict from local rank
        rank: Current rank
        world_size: Total number of ranks
    
    Returns:
        List of summaries from all ranks (only on rank 0, None otherwise)
    """
    # Convert summary to tensor for gathering
    forward_mean = torch.tensor([local_summary['forward_mean']], dtype=torch.float32)
    forward_p95 = torch.tensor([local_summary['forward_p95']], dtype=torch.float32)
    
    if rank == 0:
        # Gather from all ranks
        forward_means = [torch.zeros(1) for _ in range(world_size)]
        forward_p95s = [torch.zeros(1) for _ in range(world_size)]
        
        dist.gather(forward_mean, forward_means, dst=0)
        dist.gather(forward_p95, forward_p95s, dst=0)
        
        # Calculate variance across ranks
        means_array = torch.stack(forward_means).squeeze().numpy()
        variance_across_ranks = float(np.var(means_array))
        cv = float(np.std(means_array) / np.mean(means_array))
        
        return {
            'forward_mean_per_rank': means_array.tolist(),
            'variance_across_ranks': variance_across_ranks,
            'coefficient_of_variation': cv,
        }
    else:
        dist.gather(forward_mean, dst=0)
        dist.gather(forward_p95, dst=0)
        return None


def run_experiment(model, dataset, config, metrics_collector):
    """
    Common experiment logic: warmup + benchmark.
    
    Args:
        model: LLM model
        dataset: Dataset instance
        config: Configuration dictionary
        metrics_collector: MetricsCollector instance
    
    Returns:
        MetricsCollector with recorded metrics
    """
    warmup_iters = config['execution']['warmup_iterations']
    benchmark_iters = config['execution']['benchmark_iterations']
    
    # Warmup phase
    for i in range(warmup_iters):
        batch = dataset.get_batch()
        with Timer() as t:
            _ = model(batch)
        metrics_collector.record_warmup_time(t.elapsed)
    
    # Benchmark phase
    for i in range(benchmark_iters):
        batch = dataset.get_batch()
        
        with Timer() as t:
            output = model(batch)
        
        metrics_collector.record_forward_time(t.elapsed)
    
    return metrics_collector


def print_summary(summary, backend_name, rank):
    """
    Print experiment summary.
    
    Args:
        summary: Summary dictionary
        backend_name: Name of backend
        rank: Current rank
    """
    if rank == 0:
        print("\n" + "="*60)
        print(f"RESULTS - {backend_name}")
        print("="*60)
        print(f"Initialization time: {summary['init_time']:.4f}s")
        print(f"Forward pass mean:   {summary['forward_mean']:.4f}s")
        print(f"Forward pass std:    {summary['forward_std']:.4f}s")
        print(f"Forward pass p95:    {summary['forward_p95']:.4f}s")
        print(f"Forward pass p99:    {summary['forward_p99']:.4f}s")
        print("="*60)


def save_results(results, output_path, rank):
    """
    Save results to JSON file (only rank 0).
    
    Args:
        results: Results dictionary
        output_path: Output file path
        rank: Current rank
    """
    if rank == 0:
        save_json(results, output_path)
        print(f"\nResults saved to: {output_path}")

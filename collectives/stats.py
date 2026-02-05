"""
Statistics Calculation Script
Calculates statistics from MPI benchmark JSON files
"""

import json
import numpy as np
import os
from pathlib import Path
import csv

# ============================================================================
# HYPERPARAMETERS (HARDCODED)
# ============================================================================

# Input directory containing JSON files
INPUT_DIR = "results/openmpi"

# Output directory for statistics
OUTPUT_DIR = "stats/openmpi"

# Output CSV filename
OUTPUT_CSV = "benchmark_statistics.csv"

# ============================================================================
# STATISTICS CALCULATION FUNCTIONS
# ============================================================================

def calculate_statistics(timings_2d):
    """
    Calculate statistics from timing data
    
    Args:
        timings_2d: List of lists [rank][iteration] containing timing values
    
    Returns:
        Dictionary containing all statistics
    """
    # Convert to numpy array for easier computation
    timings_array = np.array(timings_2d)  # Shape: (num_ranks, num_iterations)
    
    # Per-rank mean (averaged across iterations)
    per_rank_means = np.mean(timings_array, axis=1)
    
    # Flatten all timings for overall statistics
    all_timings = timings_array.flatten()
    
    # Calculate aggregate statistics across all ranks and iterations
    mean_time = np.mean(all_timings)
    median_time = np.median(all_timings)
    min_time = np.min(all_timings)
    max_time = np.max(all_timings)
    std_dev = np.std(all_timings)
    percentile_95 = np.percentile(all_timings, 95)
    percentile_99 = np.percentile(all_timings, 99)
    
    # Calculate load imbalance using max of per-rank means
    max_rank_mean = np.max(per_rank_means)
    mean_of_rank_means = np.mean(per_rank_means)
    
    if mean_of_rank_means > 0:
        load_imbalance = ((max_rank_mean - mean_of_rank_means) / mean_of_rank_means) * 100
    else:
        load_imbalance = 0.0
    
    stats = {
        "mean_time_us": mean_time * 1e6,  # Convert to microseconds
        "median_time_us": median_time * 1e6,
        "min_time_us": min_time * 1e6,
        "max_time_us": max_time * 1e6,
        "std_dev_us": std_dev * 1e6,
        "p95_time_us": percentile_95 * 1e6,
        "p99_time_us": percentile_99 * 1e6,
        "load_imbalance_percent": load_imbalance,
        "per_rank_means_us": (per_rank_means * 1e6).tolist()
    }
    
    return stats

def calculate_bandwidth(num_elements, dtype, time_seconds, operation, num_ranks):
    """
    Calculate bandwidth for data transfer operations
    
    Args:
        num_elements: Number of elements in array
        dtype: Data type (for calculating bytes)
        time_seconds: Time in seconds (use max time)
        operation: Name of the operation
        num_ranks: Number of MPI ranks
    
    Returns:
        Bandwidth in GB/s, or None if not applicable
    """
    # Size of fp16 in bytes
    element_size = 2  # fp16 = 2 bytes
    
    # Calculate total data transferred based on operation
    # These are approximate logical data volumes
    data_volume_bytes = 0
    
    if operation == "allreduce":
        # Each rank sends and receives N elements
        data_volume_bytes = num_elements * element_size * num_ranks
    elif operation == "allgather":
        # Each rank sends N elements, receives N*P elements
        data_volume_bytes = num_elements * element_size * num_ranks
    elif operation == "broadcast":
        # Root sends N elements to all ranks
        data_volume_bytes = num_elements * element_size * num_ranks
    elif operation == "gather":
        # All ranks send N elements to root
        data_volume_bytes = num_elements * element_size * num_ranks
    elif operation == "scatter":
        # Root sends N elements to each rank
        data_volume_bytes = num_elements * element_size * num_ranks
    elif operation == "reduce":
        # All ranks send N elements to root
        data_volume_bytes = num_elements * element_size * num_ranks
    elif operation == "alltoall":
        # Each rank sends N/P to each rank, receives N/P from each rank
        data_volume_bytes = num_elements * element_size * num_ranks
    elif operation == "sendrecv":
        # Each rank sends and receives N elements (ring pattern)
        data_volume_bytes = num_elements * element_size * num_ranks
    else:
        return None
    
    if time_seconds > 0:
        bandwidth_gbps = (data_volume_bytes / time_seconds) / (1024**3)  # GB/s
        return bandwidth_gbps
    else:
        return None

# ============================================================================
# MAIN PROCESSING LOGIC
# ============================================================================

def process_all_json_files():
    """
    Process all JSON files in INPUT_DIR and calculate statistics
    """
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Get all JSON files
    json_files = list(Path(INPUT_DIR).glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in '{INPUT_DIR}' directory")
        return
    
    print(f"Found {len(json_files)} JSON files")
    print(f"Processing...")
    print("-" * 80)
    
    # Store all results for CSV output
    all_results = []
    
    # Process each JSON file
    for json_file in sorted(json_files):
        print(f"Processing: {json_file.name}")
        
        try:
            # Load JSON data
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract metadata
            mpi_impl = data["mpi_implementation"]
            operation = data["operation"]
            num_ranks = data["num_ranks"]
            data_size_name = data["data_size_name"]
            num_elements = data["num_elements"]
            dtype = data["dtype"]
            timings_2d = data["timings"]
            
            # Calculate statistics
            stats = calculate_statistics(timings_2d)
            
            # Calculate bandwidth (use max time for conservative estimate)
            max_time_seconds = stats["max_time_us"] / 1e6
            bandwidth = calculate_bandwidth(
                num_elements, 
                dtype, 
                max_time_seconds, 
                operation, 
                num_ranks
            )
            
            if bandwidth is not None:
                stats["bandwidth_gbps"] = bandwidth
            else:
                stats["bandwidth_gbps"] = None
            
            # Create output dictionary
            result = {
                "mpi_implementation": mpi_impl,
                "operation": operation,
                "num_ranks": num_ranks,
                "data_size_name": data_size_name,
                "num_elements": num_elements,
                "dtype": dtype,
                **stats
            }
            
            # Save individual statistics JSON
            output_filename = json_file.stem + "_stats.json"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Add to results list for CSV
            all_results.append(result)
            
        except Exception as e:
            print(f"  ERROR processing {json_file.name}: {e}")
            continue
    
    print("-" * 80)
    print(f"Processed {len(all_results)} files successfully")
    
    # Create consolidated CSV file
    if all_results:
        csv_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV)
        
        # Define CSV columns
        csv_columns = [
            "mpi_implementation",
            "operation",
            "num_ranks",
            "data_size_name",
            "num_elements",
            "mean_time_us",
            "median_time_us",
            "min_time_us",
            "max_time_us",
            "std_dev_us",
            "p95_time_us",
            "p99_time_us",
            "load_imbalance_percent",
            "bandwidth_gbps"
        ]
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            
            for result in all_results:
                # Create row without per_rank_means (too detailed for CSV)
                row = {k: v for k, v in result.items() if k != "per_rank_means_us" and k != "dtype"}
                writer.writerow(row)
        
        print(f"Consolidated CSV saved: {csv_path}")
        print(f"Individual statistics JSONs saved in: {OUTPUT_DIR}/")
        
        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total configurations processed: {len(all_results)}")
        print(f"MPI implementations: {set(r['mpi_implementation'] for r in all_results)}")
        print(f"Operations: {set(r['operation'] for r in all_results)}")
        print(f"Rank counts: {sorted(set(r['num_ranks'] for r in all_results))}")
        print(f"Data sizes: {sorted(set(r['data_size_name'] for r in all_results), key=lambda x: all_results[0]['num_elements'] if x == all_results[0]['data_size_name'] else 0)}")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    process_all_json_files()
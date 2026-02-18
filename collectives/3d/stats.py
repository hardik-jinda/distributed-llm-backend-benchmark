"""
Statistics Calculation Script for 3D Tensor Benchmarks
Calculates statistics from 3D tensor benchmark JSON files
Processes ONE implementation at a time
Generates two CSV formats: one for easy column copying, one for analysis
CSV organized by: operation → ranks → hidden_dim → seq_len → batch (varies)
"""

import json
import numpy as np
import os
from pathlib import Path
import csv
import sys

# Hyperparameters - CHOOSE ONE AT A TIME
IMPLEMENTATION = "openmpi"  # Change to: "openmpi", "intelmpi", "deepspeed_gloo", "deepspeed_oneccl"

INPUT_DIRS = {
    'openmpi': 'results_3d_tensors',
    'intelmpi': 'results_3d_tensors',
    'deepspeed_gloo': 'results_3d_tensors_deepspeed_gloo',
    'deepspeed_oneccl': 'results_3d_tensors_deepspeed_oneccl'
}

OUTPUT_DIR = f"statistics_3d_tensors_{IMPLEMENTATION}"

# Output CSV filenames
OUTPUT_CSV_TRANSPOSE = f"benchmark_statistics_3d_{IMPLEMENTATION}_transpose.csv"
OUTPUT_CSV_STANDARD = f"benchmark_statistics_3d_{IMPLEMENTATION}_standard.csv"

def calculate_statistics(timings_2d):
    """Calculate statistics from timing data"""
    timings_array = np.array(timings_2d)
    all_timings = timings_array.flatten()
    
    mean_time = np.mean(all_timings)
    median_time = np.median(all_timings)
    min_time = np.min(all_timings)
    max_time = np.max(all_timings)
    
    stats = {
        "mean_time_ms": mean_time * 1e3,
        "median_time_ms": median_time * 1e3,
        "min_time_ms": min_time * 1e3,
        "max_time_ms": max_time * 1e3
    }
    
    return stats

def process_implementation():
    """Process all JSON files for the selected implementation"""
    
    if IMPLEMENTATION not in INPUT_DIRS:
        print(f"Error: Unknown implementation '{IMPLEMENTATION}'")
        print(f"Available: {list(INPUT_DIRS.keys())}")
        sys.exit(1)
    
    input_dir = INPUT_DIRS[IMPLEMENTATION]
    
    if not Path(input_dir).exists():
        print(f"Error: Directory not found: {input_dir}")
        sys.exit(1)
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Get all JSON files
    json_files = list(Path(input_dir).glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        sys.exit(1)
    
    print(f"Processing {IMPLEMENTATION}")
    print(f"Input directory: {input_dir}")
    print(f"Found {len(json_files)} JSON files")
    print("-" * 80)
    
    all_results = []
    
    for json_file in sorted(json_files):
        print(f"Processing: {json_file.name}")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract metadata
            impl_name = data.get("mpi_implementation") or data.get("implementation")
            operation = data["operation"]
            num_ranks = data["num_ranks"]
            batch = data["tensor_shape"]["batch"]
            seq_len = data["tensor_shape"]["seq_len"]
            hidden_dim = data["tensor_shape"]["hidden_dim"]
            tensor_size_mb = data["tensor_size_mb"]
            num_elements = data["num_elements"]
            timings_2d = data["timings"]
            
            # Calculate statistics
            stats = calculate_statistics(timings_2d)
            
            # Create output dictionary
            result = {
                "implementation": impl_name,
                "operation": operation,
                "num_ranks": num_ranks,
                "hidden_dim": hidden_dim,
                "seq_len": seq_len,
                "batch": batch,
                "tensor_size_mb": round(tensor_size_mb, 4),
                "num_elements": num_elements,
                **stats
            }
            
            # Save individual statistics JSON
            output_filename = json_file.stem + "_stats.json"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            all_results.append(result)
            
        except Exception as e:
            print(f"  ERROR processing {json_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("-" * 80)
    print(f"Processed {len(all_results)} files successfully")
    
    if not all_results:
        print("No results to export!")
        return
    
    # Generate Standard CSV
    generate_standard_csv(all_results)
    
    # Generate Transposed CSV (metrics as rows)
    generate_transpose_csv(all_results)
    
    # Print summary
    print_summary(all_results)

def generate_standard_csv(all_results):
    """Generate standard CSV format (one row per configuration)"""
    
    csv_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_STANDARD)
    
    csv_columns = [
        "implementation",
        "operation",
        "num_ranks",
        "hidden_dim",
        "seq_len",
        "batch",
        "tensor_size_mb",
        "num_elements",
        "mean_time_ms",
        "median_time_ms",
        "min_time_ms",
        "max_time_ms"
    ]
    
    # Sort by: operation → ranks → hidden_dim → seq_len → batch
    all_results_sorted = sorted(all_results, key=lambda x: (
        x['operation'],
        x['num_ranks'],
        x['hidden_dim'],
        x['seq_len'],
        x['batch']
    ))
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        
        for result in all_results_sorted:
            row = {k: result[k] for k in csv_columns}
            writer.writerow(row)
    
    print(f"\nStandard CSV saved: {csv_path}")
    print(f"  Format: One row per configuration")
    print(f"  Sorted by: operation → ranks → hidden_dim → seq_len → batch")

def generate_transpose_csv(all_results):
    """Generate transposed CSV format (metrics as rows, configs as columns)
    Organized by: operation → ranks → hidden_dim → seq_len, varying batch"""
    
    csv_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_TRANSPOSE)
    
    # Sort results by: operation → ranks → hidden_dim → seq_len → batch
    all_results_sorted = sorted(all_results, key=lambda x: (
        x['operation'],
        x['num_ranks'],
        x['hidden_dim'],
        x['seq_len'],
        x['batch']
    ))
    
    # Create a unique identifier for each configuration
    # Format: operation_rX_hXXXX_sXXXX_bXX
    for result in all_results_sorted:
        config_id = (f"{result['operation']}_"
                    f"r{result['num_ranks']}_"
                    f"h{result['hidden_dim']}_"
                    f"s{result['seq_len']}_"
                    f"b{result['batch']}")
        result['config_id'] = config_id
    
    # Get all unique config IDs (column headers)
    config_ids = [r['config_id'] for r in all_results_sorted]
    
    # Metrics to include as rows
    metrics = [
        'mean_time_ms',
        'median_time_ms',
        'min_time_ms',
        'max_time_ms'
    ]
    
    # Write transposed CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header row (config IDs)
        writer.writerow(['Metric'] + config_ids)
        
        # Write each metric as a row
        for metric in metrics:
            row = [metric]
            for result in all_results_sorted:
                row.append(result[metric])
            writer.writerow(row)
        
        # Add metadata rows for reference
        writer.writerow([])  # Empty row
        writer.writerow(['--- Metadata ---'])
        
        # Operation row
        row = ['operation']
        for result in all_results_sorted:
            row.append(result['operation'])
        writer.writerow(row)
        
        # Ranks row
        row = ['num_ranks']
        for result in all_results_sorted:
            row.append(result['num_ranks'])
        writer.writerow(row)
        
        # Hidden_dim row
        row = ['hidden_dim']
        for result in all_results_sorted:
            row.append(result['hidden_dim'])
        writer.writerow(row)
        
        # Seq_len row
        row = ['seq_len']
        for result in all_results_sorted:
            row.append(result['seq_len'])
        writer.writerow(row)
        
        # Batch row
        row = ['batch']
        for result in all_results_sorted:
            row.append(result['batch'])
        writer.writerow(row)
        
        # Tensor size row
        row = ['tensor_size_mb']
        for result in all_results_sorted:
            row.append(result['tensor_size_mb'])
        writer.writerow(row)
    
    print(f"\nTransposed CSV saved: {csv_path}")
    print(f"  Format: Metrics as rows, configurations as columns")
    print(f"  Organization: operation → ranks → hidden_dim → seq_len → batch (varying)")
    print(f"  Metric rows: {len(metrics)}")
    print(f"  Configuration columns: {len(config_ids)}")
    print(f"  Easy to copy columns for same (operation, ranks, hidden, seq_len) with different batch!")

def print_summary(all_results):
    """Print summary statistics"""
    
    operations = sorted(set(r['operation'] for r in all_results))
    ranks = sorted(set(r['num_ranks'] for r in all_results))
    batches = sorted(set(r['batch'] for r in all_results))
    seq_lens = sorted(set(r['seq_len'] for r in all_results))
    hidden_dims = sorted(set(r['hidden_dim'] for r in all_results))
    
    print("\n" + "=" * 80)
    print(f"SUMMARY - {IMPLEMENTATION}")
    print("=" * 80)
    print(f"Total configurations processed: {len(all_results)}")
    print(f"Operations: {operations}")
    print(f"Rank counts: {ranks}")
    print(f"Hidden dimensions: {hidden_dims}")
    print(f"Sequence lengths: {seq_lens}")
    print(f"Batch sizes: {batches}")
    print(f"\nOrganization: operation → ranks → hidden_dim → seq_len → batch")
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print(f"  - Individual stats JSONs: {len(all_results)} files")
    print(f"  - {OUTPUT_CSV_STANDARD}")
    print(f"  - {OUTPUT_CSV_TRANSPOSE}")
    print("=" * 80)

if __name__ == "__main__":
    process_implementation()

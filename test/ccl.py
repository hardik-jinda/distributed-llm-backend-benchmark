#!/usr/bin/env python3
"""
DeepSpeed with oneCCL Backend Test Script
Run with: mpirun -n 2 python test_deepspeed_ccl.py
"""

import torch
import oneccl_bindings_for_pytorch  # MUST import this BEFORE dist
import torch.distributed as dist
import deepspeed
import os
import sys

def init_distributed():
    """Initialize distributed environment with CCL backend"""
    try:
        # Check if running in distributed mode
        if 'RANK' not in os.environ:
            print("ERROR: Not running in distributed mode!")
            print("Please run with: mpirun -n 2 python test_deepspeed_ccl.py")
            sys.exit(1)
        
        # Initialize with CCL backend
        dist.init_process_group(backend='ccl')
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        return rank, world_size
    except Exception as e:
        print(f"Error initializing distributed: {e}")
        sys.exit(1)

def test_basic_ccl(rank, world_size):
    """Test basic CCL communication"""
    print(f"\n[Rank {rank}] Test 1: Basic CCL Communication")
    print(f"{'─'*50}")
    
    # Broadcast test
    if rank == 0:
        data = torch.tensor([100.0, 200.0, 300.0])
        print(f"[Rank {rank}] Broadcasting: {data.tolist()}")
    else:
        data = torch.zeros(3)
    
    dist.broadcast(data, src=0)
    print(f"[Rank {rank}] After broadcast: {data.tolist()}")
    
    # All-reduce test
    tensor = torch.tensor([float(rank + 1)])
    print(f"[Rank {rank}] Before all-reduce: {tensor.item()}")
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = sum(range(1, world_size + 1))
    print(f"[Rank {rank}] After all-reduce: {tensor.item()} (expected: {expected})")
    
    assert abs(tensor.item() - expected) < 0.001, f"All-reduce failed on rank {rank}"

def test_deepspeed_ccl(rank, world_size):
    """Test DeepSpeed with CCL backend"""
    print(f"\n[Rank {rank}] Test 2: DeepSpeed with CCL Backend")
    print(f"{'─'*50}")
    
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5)
    )
    
    print(f"[Rank {rank}] Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # DeepSpeed configuration
    ds_config = {
        "train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.001
            }
        },
        "fp16": {
            "enabled": False  # CPU doesn't support FP16
        },
        "zero_optimization": {
            "stage": 2  # ZeRO stage 2 for parameter sharding
        }
    }
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    print(f"[Rank {rank}] DeepSpeed engine initialized")
    print(f"[Rank {rank}]   - Local rank: {model_engine.local_rank}")
    print(f"[Rank {rank}]   - Global rank: {model_engine.global_rank}")
    print(f"[Rank {rank}]   - World size: {model_engine.world_size}")
    
    # Training step
    print(f"[Rank {rank}] Running training step...")
    
    dummy_input = torch.randn(8, 10)
    dummy_labels = torch.randn(8, 5)
    
    output = model_engine(dummy_input)
    loss = torch.nn.functional.mse_loss(output, dummy_labels)
    
    print(f"[Rank {rank}] Loss: {loss.item():.4f}")
    
    model_engine.backward(loss)
    model_engine.step()
    
    print(f"[Rank {rank}] Training step completed successfully!")

def test_performance(rank, world_size):
    """Test CCL performance"""
    print(f"\n[Rank {rank}] Test 3: Performance Test")
    print(f"{'─'*50}")
    
    import time
    
    sizes = [1024, 10240, 102400]  # 1KB, 10KB, 100KB
    
    for size in sizes:
        data = torch.randn(size // 4)  # 4 bytes per float32
        
        dist.barrier()
        start_time = time.time()
        
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
        
        dist.barrier()
        elapsed = time.time() - start_time
        
        if rank == 0:
            bandwidth = (size / elapsed) / (1024 * 1024)  # MB/s
            print(f"  {size:>8} bytes: {elapsed*1000:>8.2f} ms, {bandwidth:>8.2f} MB/s")

def print_summary(rank, world_size):
    """Print test summary"""
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print(f"{'='*60}")
        print(f"Successfully tested DeepSpeed with oneCCL backend")
        print(f"Processes: {world_size}")
        print(f"Backend: {dist.get_backend()}")
        print(f"\nTests completed:")
        print(f"  ✓ Basic CCL communication (broadcast, all-reduce)")
        print(f"  ✓ DeepSpeed initialization with CCL")
        print(f"  ✓ DeepSpeed training step")
        print(f"  ✓ Performance benchmarks")
        print(f"\n{'='*60}")
        print(f"oneCCL backend is working perfectly! 🎉")
        print(f"{'='*60}")

def main():
    """Main test function"""
    try:
        print(f"\n{'='*60}")
        print(f"DeepSpeed with oneCCL Backend Test")
        print(f"PyTorch version: {torch.__version__}")
        print(f"DeepSpeed version: {deepspeed.__version__}")
        print(f"{'='*60}")
        
        # Initialize distributed
        rank, world_size = init_distributed()
        
        print(f"\n[Rank {rank}] Distributed initialized")
        print(f"[Rank {rank}]   - Backend: {dist.get_backend()}")
        print(f"[Rank {rank}]   - Rank: {rank}/{world_size}")
        
        # Run tests
        test_basic_ccl(rank, world_size)
        test_deepspeed_ccl(rank, world_size)
        test_performance(rank, world_size)
        
        # Final barrier
        dist.barrier()
        
        # Print summary
        print_summary(rank, world_size)
        
        # Cleanup
        dist.destroy_process_group()
        
    except Exception as e:
        rank = dist.get_rank() if dist.is_initialized() else "?"
        print(f"\n✗ [Rank {rank}] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

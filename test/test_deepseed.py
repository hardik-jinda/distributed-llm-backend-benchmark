#!/usr/bin/env python3
"""
Minimal DeepSpeed CPU Communication Test
This script tests distributed communication without importing problematic modules.
"""

import torch
import torch.distributed as dist
import os
import sys

def init_distributed():
    """Initialize distributed environment"""
    # Check if we're in a distributed environment
    if 'RANK' not in os.environ:
        print("ERROR: Not running in distributed mode!")
        print("\nPlease run with one of these commands:")
        print("  torchrun --nproc_per_node=2 test_deepspeed_minimal.py")
        print("  torchrun --nproc_per_node=4 test_deepspeed_minimal.py")
        print("  mpirun -np 2 python test_deepspeed_minimal.py")
        sys.exit(1)
    
    # Initialize process group
    dist.init_process_group(backend='ccl')  # gloo for CPU
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    return rank, world_size

def test_basic_info(rank, world_size):
    """Display basic distributed setup info"""
    print(f"\n{'='*60}")
    print(f"Process Information - Rank {rank}")
    print(f"{'='*60}")
    print(f"Rank: {rank}/{world_size}")
    print(f"World Size: {world_size}")
    print(f"Backend: {dist.get_backend()}")
    print(f"Process Group Initialized: {dist.is_initialized()}")
    
def test_broadcast(rank, world_size):
    """Test broadcast communication"""
    print(f"\n[Rank {rank}] Test 1: Broadcast")
    print(f"{'─'*50}")
    
    if rank == 0:
        data = torch.tensor([100.0, 200.0, 300.0])
        print(f"[Rank {rank}] 📤 Broadcasting: {data.tolist()}")
    else:
        data = torch.zeros(3)
        print(f"[Rank {rank}] Before: {data.tolist()}")
    
    dist.broadcast(data, src=0)
    print(f"[Rank {rank}] ✓ Received: {data.tolist()}")
    
    # Verify
    expected = torch.tensor([100.0, 200.0, 300.0])
    assert torch.allclose(data, expected), f"Broadcast failed on rank {rank}"
    
def test_all_reduce(rank, world_size):
    """Test all-reduce communication"""
    print(f"\n[Rank {rank}] Test 2: All-Reduce (Sum)")
    print(f"{'─'*50}")
    
    tensor = torch.tensor([float(rank + 1)])
    print(f"[Rank {rank}] 📤 Sending: {tensor.item()}")
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    expected_sum = sum(range(1, world_size + 1))
    print(f"[Rank {rank}] ✓ Result: {tensor.item()} (expected: {expected_sum})")
    
    # Verify
    assert abs(tensor.item() - expected_sum) < 0.001, f"All-reduce failed on rank {rank}"

def test_all_gather(rank, world_size):
    """Test all-gather communication"""
    print(f"\n[Rank {rank}] Test 3: All-Gather")
    print(f"{'─'*50}")
    
    send_tensor = torch.tensor([rank * 10.0, rank * 10.0 + 5])
    recv_tensors = [torch.zeros(2) for _ in range(world_size)]
    
    print(f"[Rank {rank}] 📤 Sending: {send_tensor.tolist()}")
    dist.all_gather(recv_tensors, send_tensor)
    
    result = [t.tolist() for t in recv_tensors]
    print(f"[Rank {rank}] ✓ Gathered: {result}")
    
    # Verify
    for i, tensor in enumerate(recv_tensors):
        expected = torch.tensor([i * 10.0, i * 10.0 + 5])
        assert torch.allclose(tensor, expected), f"All-gather failed on rank {rank}"

def test_reduce(rank, world_size):
    """Test reduce communication"""
    print(f"\n[Rank {rank}] Test 4: Reduce to Rank 0")
    print(f"{'─'*50}")
    
    reduce_tensor = torch.tensor([float(rank * 2)])
    print(f"[Rank {rank}] 📤 Sending: {reduce_tensor.item()}")
    
    dist.reduce(reduce_tensor, dst=0, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        expected = sum(i * 2 for i in range(world_size))
        print(f"[Rank {rank}] ✓ Result: {reduce_tensor.item()} (expected: {expected})")
        assert abs(reduce_tensor.item() - expected) < 0.001, "Reduce failed"
    else:
        print(f"[Rank {rank}] ✓ Reduce completed")

def test_scatter(rank, world_size):
    """Test scatter communication"""
    print(f"\n[Rank {rank}] Test 5: Scatter from Rank 0")
    print(f"{'─'*50}")
    
    if rank == 0:
        scatter_list = [torch.tensor([i * 100.0]) for i in range(world_size)]
        print(f"[Rank {rank}] 📤 Scattering: {[t.item() for t in scatter_list]}")
    else:
        scatter_list = None
    
    recv_tensor = torch.zeros(1)
    dist.scatter(recv_tensor, scatter_list, src=0)
    
    expected = rank * 100.0
    print(f"[Rank {rank}] ✓ Received: {recv_tensor.item()} (expected: {expected})")
    
    # Verify
    assert abs(recv_tensor.item() - expected) < 0.001, f"Scatter failed on rank {rank}"

def test_send_recv(rank, world_size):
    """Test point-to-point communication"""
    print(f"\n[Rank {rank}] Test 6: Point-to-Point (Send/Recv)")
    print(f"{'─'*50}")
    
    if world_size < 2:
        print(f"[Rank {rank}] ⚠ Skipped (need at least 2 ranks)")
        return
    
    if rank == 0:
        send_data = torch.tensor([42.0, 84.0])
        dist.send(send_data, dst=1)
        print(f"[Rank {rank}] 📤 Sent {send_data.tolist()} to Rank 1")
    elif rank == 1:
        recv_data = torch.zeros(2)
        dist.recv(recv_data, src=0)
        print(f"[Rank {rank}] ✓ Received {recv_data.tolist()} from Rank 0")
        
        expected = torch.tensor([42.0, 84.0])
        assert torch.allclose(recv_data, expected), "Send/Recv failed"
    else:
        print(f"[Rank {rank}] (not participating)")

def test_barrier(rank, world_size):
    """Test barrier synchronization"""
    print(f"\n[Rank {rank}] Test 7: Barrier Synchronization")
    print(f"{'─'*50}")
    
    import time
    # Simulate different arrival times
    time.sleep(rank * 0.1)
    
    print(f"[Rank {rank}] Arriving at barrier...")
    dist.barrier()
    print(f"[Rank {rank}] ✓ All ranks synchronized!")

def test_advanced_operations(rank, world_size):
    """Test more advanced collective operations"""
    print(f"\n[Rank {rank}] Test 8: Advanced Operations")
    print(f"{'─'*50}")
    
    # All-reduce with MAX operation
    tensor = torch.tensor([float(rank)])
    print(f"[Rank {rank}] Value: {tensor.item()}")
    
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    expected_max = float(world_size - 1)
    print(f"[Rank {rank}] ✓ Max across all ranks: {tensor.item()} (expected: {expected_max})")
    
    assert abs(tensor.item() - expected_max) < 0.001, f"All-reduce MAX failed on rank {rank}"

def print_summary(rank, world_size):
    """Print test summary"""
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print(f"{'='*60}")
        print(f"Successfully tested {world_size} processes on CPU")
        print(f"Backend: {dist.get_backend()}")
        print(f"\nAll communication primitives are working correctly:")
        print(f"  ✓ Broadcast")
        print(f"  ✓ All-Reduce")
        print(f"  ✓ All-Gather")
        print(f"  ✓ Reduce")
        print(f"  ✓ Scatter")
        print(f"  ✓ Send/Recv (point-to-point)")
        print(f"  ✓ Barrier")
        print(f"  ✓ Advanced operations")
        print(f"\n{'='*60}")
        print(f"Your distributed setup is working perfectly! 🎉")
        print(f"{'='*60}")

def main():
    """Main test function"""
    try:
        print(f"\n{'='*60}")
        print(f"Distributed Communication Test (CPU)")
        print(f"PyTorch version: {torch.__version__}")
        print(f"{'='*60}")
        
        # Initialize
        rank, world_size = init_distributed()
        
        # Run tests
        test_basic_info(rank, world_size)
        test_broadcast(rank, world_size)
        test_all_reduce(rank, world_size)
        test_all_gather(rank, world_size)
        test_reduce(rank, world_size)
        test_scatter(rank, world_size)
        test_send_recv(rank, world_size)
        test_barrier(rank, world_size)
        test_advanced_operations(rank, world_size)
        
        # Final barrier before summary
        dist.barrier()
        
        # Print summary
        print_summary(rank, world_size)
        
    except Exception as e:
        rank = dist.get_rank() if dist.is_initialized() else "?"
        print(f"\n✗ [Rank {rank}] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()

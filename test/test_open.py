#!/usr/bin/env python3
"""
OpenMPI Communication Test with mpi4py
This script tests OpenMPI installation and communication between ranks.
"""

import sys
import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    print("ERROR: mpi4py is not installed!")
    print("Install it with: pip install mpi4py")
    sys.exit(1)

def test_basic_info():
    """Display basic MPI setup info"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    processor_name = MPI.Get_processor_name()
    
    print(f"\n{'='*60}")
    print(f"Process Information - Rank {rank}")
    print(f"{'='*60}")
    print(f"Rank: {rank}/{size}")
    print(f"World Size: {size}")
    print(f"Processor Name: {processor_name}")
    print(f"MPI Version: {MPI.Get_version()}")
    print(f"MPI Library Version: {MPI.Get_library_version()}")
    
    return comm, rank, size

def test_send_recv(comm, rank, size):
    """Test point-to-point communication"""
    print(f"\n[Rank {rank}] Test 1: Point-to-Point (Send/Recv)")
    print(f"{'─'*50}")
    
    if size < 2:
        print(f"[Rank {rank}] ⚠ Skipped (need at least 2 ranks)")
        return
    
    if rank == 0:
        data = {'message': 'Hello from rank 0!', 'value': 42, 'array': [1, 2, 3]}
        comm.send(data, dest=1, tag=11)
        print(f"[Rank {rank}] 📤 Sent to Rank 1: {data}")
        
        # Receive response
        response = comm.recv(source=1, tag=22)
        print(f"[Rank {rank}] ✓ Received from Rank 1: {response}")
        
    elif rank == 1:
        # Receive from rank 0
        data = comm.recv(source=0, tag=11)
        print(f"[Rank {rank}] ✓ Received from Rank 0: {data}")
        
        # Send response
        response = {'message': 'Hello back from rank 1!', 'value': 84}
        comm.send(response, dest=0, tag=22)
        print(f"[Rank {rank}] 📤 Sent response to Rank 0: {response}")
    else:
        print(f"[Rank {rank}] (not participating)")

def test_broadcast(comm, rank, size):
    """Test broadcast communication"""
    print(f"\n[Rank {rank}] Test 2: Broadcast")
    print(f"{'─'*50}")
    
    if rank == 0:
        data = {'info': 'Broadcast message', 'numbers': [10, 20, 30], 'pi': 3.14159}
        print(f"[Rank {rank}] 📤 Broadcasting: {data}")
    else:
        data = None
        print(f"[Rank {rank}] Waiting for broadcast...")
    
    data = comm.bcast(data, root=0)
    
    if rank != 0:
        print(f"[Rank {rank}] ✓ Received: {data}")
    
    # Verify
    assert data['info'] == 'Broadcast message', f"Broadcast failed on rank {rank}"
    assert data['pi'] == 3.14159, f"Broadcast failed on rank {rank}"

def test_scatter(comm, rank, size):
    """Test scatter communication"""
    print(f"\n[Rank {rank}] Test 3: Scatter")
    print(f"{'─'*50}")
    
    if rank == 0:
        # Prepare different data for each rank
        data = [{'rank_id': i, 'value': i * 100} for i in range(size)]
        print(f"[Rank {rank}] 📤 Scattering: {data}")
    else:
        data = None
    
    recv_data = comm.scatter(data, root=0)
    print(f"[Rank {rank}] ✓ Received: {recv_data}")
    
    # Verify
    assert recv_data['rank_id'] == rank, f"Scatter failed on rank {rank}"
    assert recv_data['value'] == rank * 100, f"Scatter failed on rank {rank}"

def test_gather(comm, rank, size):
    """Test gather communication"""
    print(f"\n[Rank {rank}] Test 4: Gather")
    print(f"{'─'*50}")
    
    # Each rank sends its own data
    send_data = {'from_rank': rank, 'square': rank ** 2}
    print(f"[Rank {rank}] 📤 Sending: {send_data}")
    
    recv_data = comm.gather(send_data, root=0)
    
    if rank == 0:
        print(f"[Rank {rank}] ✓ Gathered from all ranks: {recv_data}")
        # Verify
        for i, data in enumerate(recv_data):
            assert data['from_rank'] == i, "Gather failed"
            assert data['square'] == i ** 2, "Gather failed"
    else:
        print(f"[Rank {rank}] ✓ Gather completed")

def test_allgather(comm, rank, size):
    """Test all-gather communication"""
    print(f"\n[Rank {rank}] Test 5: All-Gather")
    print(f"{'─'*50}")
    
    send_data = {'rank': rank, 'cube': rank ** 3}
    print(f"[Rank {rank}] 📤 Sending: {send_data}")
    
    recv_data = comm.allgather(send_data)
    print(f"[Rank {rank}] ✓ Gathered: {recv_data}")
    
    # Verify
    assert len(recv_data) == size, f"All-gather failed on rank {rank}"
    for i, data in enumerate(recv_data):
        assert data['rank'] == i, f"All-gather failed on rank {rank}"
        assert data['cube'] == i ** 3, f"All-gather failed on rank {rank}"

def test_reduce(comm, rank, size):
    """Test reduce communication"""
    print(f"\n[Rank {rank}] Test 6: Reduce (Sum)")
    print(f"{'─'*50}")
    
    send_value = rank + 1
    print(f"[Rank {rank}] 📤 Sending: {send_value}")
    
    recv_value = comm.reduce(send_value, op=MPI.SUM, root=0)
    
    if rank == 0:
        expected = sum(range(1, size + 1))
        print(f"[Rank {rank}] ✓ Reduced sum: {recv_value} (expected: {expected})")
        assert recv_value == expected, "Reduce failed"
    else:
        print(f"[Rank {rank}] ✓ Reduce completed")

def test_allreduce(comm, rank, size):
    """Test all-reduce communication"""
    print(f"\n[Rank {rank}] Test 7: All-Reduce (Sum)")
    print(f"{'─'*50}")
    
    send_value = rank * 2
    print(f"[Rank {rank}] 📤 Sending: {send_value}")
    
    recv_value = comm.allreduce(send_value, op=MPI.SUM)
    
    expected = sum(i * 2 for i in range(size))
    print(f"[Rank {rank}] ✓ All-reduced sum: {recv_value} (expected: {expected})")
    
    # Verify
    assert recv_value == expected, f"All-reduce failed on rank {rank}"

def test_numpy_arrays(comm, rank, size):
    """Test NumPy array communication"""
    print(f"\n[Rank {rank}] Test 8: NumPy Array Communication")
    print(f"{'─'*50}")
    
    if rank == 0:
        data = np.arange(10, dtype='float64')
        print(f"[Rank {rank}] 📤 Broadcasting array: {data}")
    else:
        data = np.zeros(10, dtype='float64')
    
    comm.Bcast(data, root=0)
    
    if rank != 0:
        print(f"[Rank {rank}] ✓ Received array: {data}")
    
    # Verify
    expected = np.arange(10, dtype='float64')
    assert np.allclose(data, expected), f"NumPy broadcast failed on rank {rank}"

def test_numpy_reduction(comm, rank, size):
    """Test NumPy array reduction"""
    print(f"\n[Rank {rank}] Test 9: NumPy Array Reduction")
    print(f"{'─'*50}")
    
    # Each rank creates an array with its rank value
    send_array = np.full(5, rank, dtype='float64')
    recv_array = np.zeros(5, dtype='float64')
    
    print(f"[Rank {rank}] 📤 Sending array: {send_array}")
    
    comm.Allreduce(send_array, recv_array, op=MPI.SUM)
    
    expected = np.full(5, sum(range(size)), dtype='float64')
    print(f"[Rank {rank}] ✓ Reduced array: {recv_array}")
    
    # Verify
    assert np.allclose(recv_array, expected), f"NumPy reduction failed on rank {rank}"

def test_barrier(comm, rank, size):
    """Test barrier synchronization"""
    print(f"\n[Rank {rank}] Test 10: Barrier Synchronization")
    print(f"{'─'*50}")
    
    import time
    # Simulate different arrival times
    time.sleep(rank * 0.1)
    
    print(f"[Rank {rank}] Arriving at barrier...")
    comm.Barrier()
    print(f"[Rank {rank}] ✓ All ranks synchronized!")

def test_ring_communication(comm, rank, size):
    """Test ring communication pattern"""
    print(f"\n[Rank {rank}] Test 11: Ring Communication")
    print(f"{'─'*50}")
    
    # Each rank sends to next rank in ring
    send_to = (rank + 1) % size
    recv_from = (rank - 1 + size) % size
    
    send_data = f"Message from rank {rank}"
    print(f"[Rank {rank}] 📤 Sending '{send_data}' to Rank {send_to}")
    
    # Non-blocking send/receive to avoid deadlock
    req_send = comm.isend(send_data, dest=send_to, tag=99)
    req_recv = comm.irecv(source=recv_from, tag=99)
    
    recv_data = req_recv.wait()
    req_send.wait()
    
    print(f"[Rank {rank}] ✓ Received '{recv_data}' from Rank {recv_from}")

def test_operation_variations(comm, rank, size):
    """Test different MPI operations"""
    print(f"\n[Rank {rank}] Test 12: Various MPI Operations")
    print(f"{'─'*50}")
    
    # Test MAX operation
    value = rank * 10
    max_value = comm.allreduce(value, op=MPI.MAX)
    expected_max = (size - 1) * 10
    print(f"[Rank {rank}] MAX operation: {max_value} (expected: {expected_max})")
    assert max_value == expected_max, f"MAX operation failed on rank {rank}"
    
    # Test MIN operation
    min_value = comm.allreduce(value, op=MPI.MIN)
    expected_min = 0
    print(f"[Rank {rank}] MIN operation: {min_value} (expected: {expected_min})")
    assert min_value == expected_min, f"MIN operation failed on rank {rank}"
    
    # Test PROD operation
    value = rank + 1
    prod_value = comm.allreduce(value, op=MPI.PROD)
    expected_prod = np.prod(range(1, size + 1))
    print(f"[Rank {rank}] PROD operation: {prod_value} (expected: {expected_prod})")
    assert prod_value == expected_prod, f"PROD operation failed on rank {rank}"

def print_summary(rank, size):
    """Print test summary"""
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"✓✓✓ ALL MPI TESTS PASSED! ✓✓✓")
        print(f"{'='*60}")
        print(f"Successfully tested OpenMPI with {size} processes")
        print(f"MPI Version: {MPI.Get_version()}")
        print(f"\nAll communication patterns tested:")
        print(f"  ✓ Point-to-point (Send/Recv)")
        print(f"  ✓ Broadcast")
        print(f"  ✓ Scatter")
        print(f"  ✓ Gather")
        print(f"  ✓ All-Gather")
        print(f"  ✓ Reduce")
        print(f"  ✓ All-Reduce")
        print(f"  ✓ NumPy arrays (contiguous)")
        print(f"  ✓ NumPy array reduction")
        print(f"  ✓ Barrier synchronization")
        print(f"  ✓ Ring communication pattern")
        print(f"  ✓ Various operations (MAX, MIN, PROD)")
        print(f"\n{'='*60}")
        print(f"Your OpenMPI installation is working perfectly! 🎉")
        print(f"{'='*60}")

def main():
    """Main test function"""
    try:
        # Initialize MPI
        comm, rank, size = test_basic_info()
        
        # Run all tests
        test_send_recv(comm, rank, size)
        test_broadcast(comm, rank, size)
        test_scatter(comm, rank, size)
        test_gather(comm, rank, size)
        test_allgather(comm, rank, size)
        test_reduce(comm, rank, size)
        test_allreduce(comm, rank, size)
        test_numpy_arrays(comm, rank, size)
        test_numpy_reduction(comm, rank, size)
        test_barrier(comm, rank, size)
        test_ring_communication(comm, rank, size)
        test_operation_variations(comm, rank, size)
        
        # Final barrier before summary
        comm.Barrier()
        
        # Print summary
        print_summary(rank, size)
        
    except Exception as e:
        rank = MPI.COMM_WORLD.Get_rank()
        print(f"\n✗ [Rank {rank}] Error: {e}")
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)

if __name__ == "__main__":
    print(MPI.Get_library_version())

    main()

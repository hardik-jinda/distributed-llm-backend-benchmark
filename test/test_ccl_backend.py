import torch
import torch.distributed as dist
import oneccl_bindings_for_pytorch

# Initialize with CCL backend
dist.init_process_group(backend='ccl')

rank = dist.get_rank()
world_size = dist.get_world_size()

print(f"Rank {rank}/{world_size} using CCL backend")

# Test all-reduce
tensor = torch.tensor([rank], dtype=torch.float32)
print(f"Rank {rank} before all-reduce: {tensor.item()}")

dist.all_reduce(tensor)
print(f"Rank {rank} after all-reduce: {tensor.item()}")

if rank == 0:
    print("\n✓ CCL backend is working!")

dist.destroy_process_group()

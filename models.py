"""
LLM Models with Tensor Parallelism

Implements transformer-based LLM models with tensor parallelism for distributed inference.
Supports 1B, 7B, and 13B parameter configurations.
Uses pure mpi4py for communication (no torch.distributed).
"""

import torch
import torch.nn as nn
import numpy as np
from mpi4py import MPI


# =============================================================================
# Tensor Parallel Layers
# =============================================================================

class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.
    Splits output dimension across ranks.
    """
    
    def __init__(self, in_features, out_features, rank, world_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.world_size = world_size
        
        # Each rank gets a slice of columns
        self.out_features_per_rank = out_features // world_size
        
        self.weight = nn.Parameter(
            torch.randn(in_features, self.out_features_per_rank, 
                       dtype=torch.bfloat16)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq, in_features]
        Returns:
            output: [batch, seq, out_features_per_rank]
        """
        return torch.matmul(x, self.weight)


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.
    Splits input dimension across ranks.
    Performs AllReduce after computation using mpi4py.
    """
    
    def __init__(self, in_features, out_features, rank, world_size, comm):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.world_size = world_size
        self.comm = comm
        
        # Each rank gets a slice of rows
        self.in_features_per_rank = in_features // world_size
        
        self.weight = nn.Parameter(
            torch.randn(self.in_features_per_rank, out_features,
                       dtype=torch.bfloat16)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq, in_features_per_rank]
        Returns:
            output: [batch, seq, out_features] after AllReduce
        """
        # Each rank computes partial result
        output = torch.matmul(x, self.weight)
        
        # Convert BF16 to float32 for numpy/MPI (numpy doesn't support BF16)
        output_fp32 = output.to(torch.float32)
        output_np = output_fp32.detach().cpu().numpy()
        
        # Ensure contiguous array
        if not output_np.flags['C_CONTIGUOUS']:
            output_np = output_np.copy()
        
        # Prepare output buffer
        output_reduced = np.empty_like(output_np)
        
        # AllReduce using mpi4py with contiguous arrays
        self.comm.Allreduce(output_np, output_reduced, op=MPI.SUM)
        
        # Convert back to torch tensor in BF16
        output = torch.from_numpy(output_reduced).to(dtype=torch.bfloat16)
        
        return output


# =============================================================================
# Transformer Block
# =============================================================================

class TransformerBlock(nn.Module):
    """
    Single transformer block with tensor parallelism.
    Contains: Attention + FFN with residual connections.
    """
    
    def __init__(self, hidden_size, num_heads, ffn_intermediate, rank, world_size, comm):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.rank = rank
        self.world_size = world_size
        self.comm = comm
        
        # Attention sublayer
        self.ln1 = nn.LayerNorm(hidden_size, dtype=torch.bfloat16)
        
        # QKV projection (column parallel)
        self.qkv_proj = ColumnParallelLinear(
            hidden_size, 3 * hidden_size, rank, world_size
        )
        
        # Attention output projection (row parallel)
        self.out_proj = RowParallelLinear(
            hidden_size, hidden_size, rank, world_size, comm
        )
        
        # FFN sublayer
        self.ln2 = nn.LayerNorm(hidden_size, dtype=torch.bfloat16)
        
        # FFN up projection (column parallel)
        self.ffn_up = ColumnParallelLinear(
            hidden_size, ffn_intermediate, rank, world_size
        )
        
        # FFN down projection (row parallel)
        self.ffn_down = RowParallelLinear(
            ffn_intermediate, hidden_size, rank, world_size, comm
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq, hidden_size]
        Returns:
            output: [batch, seq, hidden_size]
        """
        # Attention block
        residual = x
        x = self.ln1(x)
        
        # QKV projection (column parallel)
        # Output: [batch, seq, 3*hidden_size/world_size]
        qkv = self.qkv_proj(x)
        
        # For benchmarking purposes, we'll do simplified "attention"
        # In real attention: split qkv into Q, K, V and compute attention
        # Here: just take the first 1/3 as the attention output (simulating)
        # This gives us [batch, seq, hidden_size/world_size]
        hidden_per_rank = self.hidden_size // self.world_size
        attn_output = qkv[:, :, :hidden_per_rank]  # Take first third
        
        # Attention output projection (row parallel)
        # Input: [batch, seq, hidden_size/world_size]
        # Output: [batch, seq, hidden_size] after AllReduce
        x = self.out_proj(attn_output)
        x = x + residual
        
        # FFN block
        residual = x
        x = self.ln2(x)
        
        # FFN up projection (column parallel)
        # Output: [batch, seq, ffn_intermediate/world_size]
        x = self.ffn_up(x)
        x = torch.nn.functional.gelu(x)
        
        # FFN down projection (row parallel)
        # Input: [batch, seq, ffn_intermediate/world_size]
        # Output: [batch, seq, hidden_size] after AllReduce
        x = self.ffn_down(x)
        x = x + residual
        
        return x


# =============================================================================
# Main LLM Model
# =============================================================================

class LLM(nn.Module):
    """
    Transformer-based LLM with tensor parallelism.
    Configurable for different model sizes.
    """
    
    def __init__(self, hidden_size, num_layers, num_heads, ffn_intermediate, 
                 rank, world_size, comm):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_intermediate = ffn_intermediate
        self.rank = rank
        self.world_size = world_size
        self.comm = comm
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ffn_intermediate, 
                           rank, world_size, comm)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_final = nn.LayerNorm(hidden_size, dtype=torch.bfloat16)
    
    def forward(self, x):
        """
        Forward pass through all transformer layers.
        
        Args:
            x: [batch, seq, hidden_size] in BF16
        Returns:
            output: [batch, seq, hidden_size] in BF16
        """
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_final(x)
        return x
    
    def get_num_parameters(self):
        """Return total number of parameters across all ranks."""
        return sum(p.numel() for p in self.parameters()) * self.world_size
    
    def get_memory_footprint(self):
        """Return model memory footprint in bytes (per rank)."""
        return sum(p.numel() * p.element_size() for p in self.parameters())


# =============================================================================
# Model Configurations
# =============================================================================

MODEL_CONFIGS = {
    '1B': {
        'hidden_size': 2048,
        'num_layers': 24,
        'num_heads': 16,
        'ffn_intermediate': 8192,
    },
    '7B': {
        'hidden_size': 4096,
        'num_layers': 32,
        'num_heads': 32,
        'ffn_intermediate': 16384,
    },
    '13B': {
        'hidden_size': 5120,
        'num_layers': 40,
        'num_heads': 40,
        'ffn_intermediate': 20480,
    }
}


# =============================================================================
# Model Factory Function
# =============================================================================

def create_model(model_size, rank, world_size, comm):
    """
    Create LLM model with specified size.
    
    Args:
        model_size: "1B", "7B", or "13B"
        rank: Process rank
        world_size: Total number of processes
        comm: MPI communicator
    
    Returns:
        LLM model instance
    """
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Invalid model size: {model_size}. "
                        f"Choose from {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_size]
    
    model = LLM(
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ffn_intermediate=config['ffn_intermediate'],
        rank=rank,
        world_size=world_size,
        comm=comm
    )
    
    return model


def create_model_from_config(config, rank, world_size, comm):
    """
    Create model from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        rank: Process rank
        world_size: Total number of processes
        comm: MPI communicator
    
    Returns:
        LLM model instance
    """
    model_config = config['model']
    
    model = LLM(
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        ffn_intermediate=model_config['ffn_intermediate'],
        rank=rank,
        world_size=world_size,
        comm=comm
    )
    
    return model

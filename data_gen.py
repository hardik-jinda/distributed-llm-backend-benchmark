"""
Dataset Generation Module

Generates fixed synthetic BF16 embeddings for LLM inference benchmarking.
"""

import torch


class SyntheticEmbeddingDataset:
    """
    Generate fixed synthetic BF16 embeddings for tensor parallel LLM inference.
    """
    
    def __init__(
        self,
        batch_size: int,
        sequence_length: int,
        hidden_size: int,
        seed: int = 42
    ):
        """
        Initialize dataset generator.
        
        Args:
            batch_size: Number of samples per batch
            sequence_length: Number of tokens per sequence
            hidden_size: Hidden dimension size
            seed: Random seed for reproducibility
        """
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.seed = seed
        
        # Generate fixed batch
        torch.manual_seed(self.seed)
        self.fixed_batch = torch.randn(
            self.batch_size,
            self.sequence_length,
            self.hidden_size,
            dtype=torch.bfloat16,
            device='cpu'
        )
    
    def get_batch(self) -> torch.Tensor:
        """
        Get the fixed batch of embeddings.
        
        Returns:
            Tensor of shape [batch_size, sequence_length, hidden_size] in BF16
        """
        return self.fixed_batch


def create_dataset_from_config(config: dict) -> SyntheticEmbeddingDataset:
    """
    Create dataset from configuration dictionary.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        SyntheticEmbeddingDataset instance
    """
    dataset = SyntheticEmbeddingDataset(
        batch_size=config['input']['batch_size'],
        sequence_length=config['input']['sequence_length'],
        hidden_size=config['model']['hidden_size'],
        seed=config['input']['seed']
    )
    
    return dataset

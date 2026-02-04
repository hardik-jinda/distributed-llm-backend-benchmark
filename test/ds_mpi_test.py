import torch
import deepspeed
import torch.distributed as dist

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

ds_config = {
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1,
    "zero_optimization": {
        "stage": 0
    },
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-3
        }
    }
}

def main():
    dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Hello from rank {rank} / {world_size}")

    model = DummyModel()

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    x = torch.randn(1, 10)
    y = model_engine(x)
    loss = y.mean()

    model_engine.backward(loss)
    model_engine.step()

    print(f"Rank {rank}: step completed")

if __name__ == "__main__":
    main()


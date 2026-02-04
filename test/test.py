import torch
import deepspeed

# Simple dummy model
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

model = DummyModel()

# Minimal DeepSpeed config for CPU
ds_config = {
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "fp16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 0
    }
}

# Initialize DeepSpeed engine
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)

# Dummy input
x = torch.randn(1, 10)
y = model_engine(x)
loss = y.mean()

# Backward + step
model_engine.backward(loss)
model_engine.step()

print("SUCCESS: DeepSpeed initialized and ran a training step on CPU")


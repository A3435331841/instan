import torch

from panotrack.models.spherical_memory import SphericalMemoryNet


def test_spherical_memory_forward_and_backward():
    torch.manual_seed(0)
    model = SphericalMemoryNet(channels=8)
    template = torch.rand(2, 3, 32, 32)
    search = torch.rand(2, 3, 64, 64)
    geometry = torch.tensor([[179.0, 65.0, 60.0, 45.0],
                             [-179.0, -65.0, 60.0, 45.0]])
    out = model(template, search, geometry)
    assert out["delta"].shape == (2, 4)
    assert out["confidence"].shape == (2,)
    loss = model.loss(out, torch.zeros(2, 4))["total"]
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_memory_freezes_low_confidence():
    model = SphericalMemoryNet(channels=8)
    memory = torch.ones(1, 8, 8, 8)
    current = torch.zeros_like(memory)
    updated = model.update_memory(memory, current, torch.tensor([0.1]))
    assert torch.allclose(updated, memory)

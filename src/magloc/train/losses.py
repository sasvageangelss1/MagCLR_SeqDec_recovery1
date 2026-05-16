from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    b = z1.size(0)
    sim = z @ z.t() / temperature
    sim = sim.masked_fill(torch.eye(2 * b, dtype=torch.bool, device=z.device), float("-inf"))
    labels = torch.arange(b, device=z.device)
    labels = torch.cat([labels + b, labels], dim=0)
    return F.cross_entropy(sim, labels)

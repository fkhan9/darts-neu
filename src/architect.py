"""
Architect: handles the architecture-parameter (alpha) update step of
DARTS's bilevel optimization.

Uses the FIRST-ORDER approximation (xi=0): we compute the gradient of
validation loss w.r.t. alpha, treating current network weights w as
fixed (not the more accurate but slower second-order finite-difference
correction). Deliberate speed/accuracy tradeoff given time constraints.
"""

import torch


class Architect:
    def __init__(self, model, arch_lr=3e-4, arch_weight_decay=1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.arch_parameters(),
            lr=arch_lr,
            betas=(0.5, 0.999),
            weight_decay=arch_weight_decay,
        )

    def step(self, x_val, y_val, criterion):
        self.optimizer.zero_grad()
        logits = self.model(x_val)
        loss = criterion(logits, y_val)
        loss.backward()
        self.optimizer.step()
        return loss.item()
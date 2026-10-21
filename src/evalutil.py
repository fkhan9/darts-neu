import numpy as np
import torch


def predict(model, loader, device):
    """Logits and labels for every sample in loader (model put in eval mode, then back to train)."""
    model.eval()
    outs, labs = [], []
    with torch.no_grad():
        for x, y in loader:
            outs.append(model(x.to(device)).float().cpu().numpy())
            labs.append(y.numpy())
    model.train()
    return np.concatenate(outs), np.concatenate(labs)

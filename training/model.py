"""The HW2 classifier: a small multi-layer perceptron over a region's activation vector.

Input:  one 3584-d vector, Qwen2.5-VL-7B's hidden state 21 mean-pooled over the tokens under a box.
Output: logits over 20 object classes.

hidden=() gives softmax regression (no hidden layer), the shallow baseline.
The input standardization (training mean and std) lives inside the model as buffers,
so a saved checkpoint carries it and serving applies exactly the same transform.
"""
from __future__ import annotations

import torch
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


class MLP(nn.Module):
    def __init__(self, d_in: int, n_classes: int, hidden: tuple[int, ...] = (512,), activation: str = "relu", dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        d = d_in
        for h in hidden:
            layers += [nn.Linear(d, h), ACTIVATIONS[activation](), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)
        self.register_buffer("mean", torch.zeros(d_in))
        self.register_buffer("std", torch.ones(d_in))
        self.config = {"d_in": d_in, "n_classes": n_classes, "hidden": list(hidden), "activation": activation, "dropout": dropout}
        self.init_weights(activation)

    def init_weights(self, activation: str) -> None:
        """Step 1 of training: He (Kaiming) init for ReLU-family layers, Xavier otherwise; zero biases."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                if activation in ("relu", "gelu"):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def set_standardization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean.copy_(mean)
        self.std.copy_(std.clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)


def load_checkpoint(path, map_location="cpu") -> tuple[MLP, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    c = ckpt["config"]
    model = MLP(c["d_in"], c["n_classes"], tuple(c["hidden"]), c["activation"], c["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt

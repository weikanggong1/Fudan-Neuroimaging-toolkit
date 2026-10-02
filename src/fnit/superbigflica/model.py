"""Independent implementation of the published supervised linked-component model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SupervisedComponents(nn.Module):
    """Shared sparse spatial encoder, tied reconstruction and phenotype heads."""

    def __init__(self, n_features: list[int], n_components: int,
                 output_sizes: list[int], dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout
        self.output_sizes = output_sizes
        self.spatial = nn.ParameterList([
            nn.Parameter(torch.randn(size, n_components)) for size in n_features])
        self.modality_logits = nn.Parameter(torch.ones(n_components, len(n_features)))
        self.prediction_weight = nn.Parameter(torch.randn(n_components, sum(output_sizes)))
        self.prediction_bias = nn.Parameter(torch.randn(sum(output_sizes)))
        self.normalization = nn.BatchNorm1d(n_components)

    def forward(self, images: list[torch.Tensor], *, reconstruct: bool = True):
        weights = self.modality_logits.softmax(dim=1)
        encoded = [F.dropout(image, self.dropout, self.training) @ spatial * weights[:, i]
                   for i, (image, spatial) in enumerate(zip(images, self.spatial))]
        latent = self.normalization(torch.stack(encoded).sum(dim=0) / len(images))
        latent = F.dropout(latent, self.dropout, self.training)
        predictions = latent @ self.prediction_weight + self.prediction_bias
        rebuilt = [(latent * weights[:, i]) @ spatial.T
                   for i, spatial in enumerate(self.spatial)] if reconstruct else []
        return rebuilt, latent, predictions


class SupervisedObjective(nn.Module):
    """Reconstruction, spatial sparsity, supervision and prediction regularization.

    Positive scales are squared raw parameters. The continuous complete-label
    objective agrees with the fixed upstream equations; missing labels average
    over observed entries, and categorical heads use training-weighted cross entropy.
    """

    def __init__(self, n_modalities: int, targets: list[dict],
                 output_sizes: list[int], relative_weight: float = 0.5):
        super().__init__()
        self.targets, self.output_sizes = targets, output_sizes
        for index, (target, width) in enumerate(zip(targets, output_sizes)):
            if target['type'] == 'categorical':
                self.register_buffer(f'class_weights_{index}', torch.tensor(
                    target.get('class_weights', [1.] * width), dtype=torch.float32))
        self.scales = nn.ParameterList([nn.Parameter(torch.ones(size)) for size in
                                        (n_modalities, n_modalities, len(targets),
                                         len(targets), len(targets))])
        self.register_buffer('balance', torch.tensor([
            relative_weight, relative_weight, 1 - relative_weight,
            1 - relative_weight]) / 2)

    def forward(self, model: SupervisedComponents, images: list[torch.Tensor],
                rebuilt: list[torch.Tensor], predictions: torch.Tensor,
                labels: torch.Tensor, training_size: int):
        scales = [value.square().clamp_min(1e-6) for value in self.scales]
        proportion = labels.shape[0] / training_size
        reconstruction = sum((estimate - image).square().mean() /
                             (2 * scales[0][i].square())
                             for i, (estimate, image) in enumerate(zip(rebuilt, images)))
        reconstruction = reconstruction + torch.log1p(scales[0]).sum()
        sparsity = sum(proportion * loading.abs().mean() / scales[1][i]
                       for i, loading in enumerate(model.spatial))
        sparsity = sparsity + 2 * torch.log1p(scales[1]).sum()
        supervision = predictions.sum() * 0
        regularization = predictions.sum() * 0
        offset = 0
        for i, (target, width) in enumerate(zip(self.targets, self.output_sizes)):
            observed = torch.isfinite(labels[:, i])
            head = predictions[:, offset:offset + width]
            if bool(observed.any()):
                if target['type'] == 'continuous':
                    task_loss = (head[observed, 0] - labels[observed, i]).square().mean()
                    supervision = supervision + task_loss / (2 * scales[2][i].square())
                else:
                    task_loss = F.cross_entropy(head[observed], labels[observed, i].long(),
                                                weight=getattr(self, f'class_weights_{i}'))
                    supervision = supervision + task_loss / scales[2][i]
                supervision = supervision + torch.log1p(scales[2][i])
            weights = model.prediction_weight[:, offset:offset + width]
            regularization = regularization + proportion * (
                weights.abs().mean() / scales[3][i] +
                weights.square().mean() / (2 * scales[4][i].square()))
            regularization = regularization + 2 * torch.log1p(scales[3][i])
            regularization = regularization + torch.log1p(scales[4][i])
            offset += width
        # The upstream complete-label MSE averages over all tasks, while its
        # scale log terms sum over tasks. Preserve that distinction.
        log_supervision = sum(torch.log1p(scales[2][i]) for i in range(len(self.targets))
                              if bool(torch.isfinite(labels[:, i]).any()))
        supervision = (supervision - log_supervision) / len(self.targets) + log_supervision
        log_regularization = 2 * torch.log1p(scales[3]).sum() + torch.log1p(scales[4]).sum()
        regularization = ((regularization - log_regularization) / len(self.targets) +
                          log_regularization)
        terms = torch.stack([reconstruction, sparsity, supervision, regularization])
        return (terms * self.balance).sum(), terms

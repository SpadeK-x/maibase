from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class MLPConfig:
    input_dim: int = 84
    event_hidden_dim: int = 128
    event_embed_dim: int = 64
    classifier_hidden_dim: int = 128
    num_classes: int = 4
    dropout: float = 0.2
    pooling: str = "mean"


class EventMLPEncoder(nn.Module):
    """
    Encodes one event vector from the fixed MVP feature space into a dense embedding.

    Input:
        x: [B, T, F] or [N, F]
    Output:
        h: same leading shape, final dim = event_embed_dim
    """

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.event_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.event_hidden_dim, config.event_embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EventPooler(nn.Module):
    """
    Pools event embeddings over the time dimension.

    Expected input:
        x: [B, T, D]
        mask: [B, T], True means valid event, False means padding
    """

    def __init__(self, pooling: str = "mean") -> None:
        super().__init__()
        if pooling not in {"mean", "max", "mean_max"}:
            raise ValueError(f"Unsupported pooling mode: {pooling}")
        self.pooling = pooling

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape [B, T, D], got {tuple(x.shape)}")

        batch_size, seq_len, dim = x.shape
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
        if mask.shape != (batch_size, seq_len):
            raise ValueError(f"Expected mask shape {(batch_size, seq_len)}, got {tuple(mask.shape)}")

        valid_mask = mask.unsqueeze(-1)  # [B, T, 1]

        if self.pooling == "mean":
            return self.masked_mean(x, valid_mask)

        if self.pooling == "max":
            return self.masked_max(x, valid_mask)

        mean_pooled = self.masked_mean(x, valid_mask)
        max_pooled = self.masked_max(x, valid_mask)
        return torch.cat([mean_pooled, max_pooled], dim=-1)

    @staticmethod
    def masked_mean(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        masked_x = x * valid_mask
        denom = valid_mask.sum(dim=1).clamp_min(1)
        return masked_x.sum(dim=1) / denom

    @staticmethod
    def masked_max(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        neg_inf = torch.finfo(x.dtype).min
        masked_x = x.masked_fill(~valid_mask, neg_inf)
        pooled = masked_x.max(dim=1).values
        all_pad = ~valid_mask.any(dim=1)
        pooled = torch.where(all_pad, torch.zeros_like(pooled), pooled)
        return pooled


class MVPMLPClassifier(nn.Module):
    """
    Minimal prediction framework for event-based chart difficulty classification.

    Expected inputs:
        x: [B, T, 84]
        mask: [B, T], True = valid event, False = padding

    Output:
        logits: [B, num_classes]
        chart_embedding: [B, D]
        event_embeddings: [B, T, event_embed_dim]
    """

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config = config
        self.event_encoder = EventMLPEncoder(config)
        self.pooler = EventPooler(config.pooling)

        pooled_dim = config.event_embed_dim
        if config.pooling == "mean_max":
            pooled_dim *= 2

        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape [B, T, F], got {tuple(x.shape)}")
        if x.size(-1) != self.config.input_dim:
            raise ValueError(
                f"Expected feature dim {self.config.input_dim}, got {x.size(-1)}"
            )

        event_embeddings = self.event_encoder(x)
        chart_embedding = self.pooler(event_embeddings, mask)
        logits = self.classifier(chart_embedding)
        return logits, chart_embedding, event_embeddings


def build_model(
    input_dim: int = 84,
    num_classes: int = 4,
    pooling: str = "mean",
) -> MVPMLPClassifier:
    config = MLPConfig(
        input_dim=input_dim,
        num_classes=num_classes,
        pooling=pooling,
    )
    return MVPMLPClassifier(config)


def make_padding_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """
    Converts sequence lengths into a boolean valid-event mask.

    Input:
        lengths: [B]
    Output:
        mask: [B, T], True = valid event
    """

    if lengths.dim() != 1:
        raise ValueError(f"Expected lengths to have shape [B], got {tuple(lengths.shape)}")
    if max_len is None:
        max_len = int(lengths.max().item())
    steps = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return steps < lengths.unsqueeze(1)


def smoke_test() -> None:
    model = build_model()
    batch_size = 3
    seq_len = 20
    feature_dim = 84
    x = torch.randn(batch_size, seq_len, feature_dim)
    lengths = torch.tensor([20, 13, 7])
    mask = make_padding_mask(lengths, seq_len)

    logits, chart_embedding, event_embeddings = model(x, mask)
    print("logits:", tuple(logits.shape))
    print("chart_embedding:", tuple(chart_embedding.shape))
    print("event_embeddings:", tuple(event_embeddings.shape))


if __name__ == "__main__":
    smoke_test()

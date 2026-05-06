import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    input_dim: int = 89
    model_dim: int = 128
    ff_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.2
    num_classes: int = 4
    max_len: int = 4096
    pooling: str = "cls"


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / model_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, T, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class MVPTransformerClassifier(nn.Module):
    """
    Transformer-based chart classifier.

    Input:
        x: [B, T, INPUT_DIM]
        mask: [B, T], True = valid event, False = padding

    Output:
        logits: [B, num_classes]
        chart_embedding: [B, D]
        token_embeddings: [B, T+1, D]
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        if config.pooling not in {"cls", "cls_mean"}:
            raise ValueError(f"Unsupported pooling mode: {config.pooling}")

        self.config = config
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.model_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.pos_encoder = SinusoidalPositionalEncoding(config.model_dim, config.max_len + 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.model_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.model_dim)

        classifier_in_dim = config.model_dim if config.pooling == "cls" else config.model_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape [B, T, F], got {tuple(x.shape)}")
        if x.size(-1) != self.config.input_dim:
            raise ValueError(f"Expected feature dim {self.config.input_dim}, got {x.size(-1)}")

        batch_size, seq_len, _ = x.shape
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)

        h = self.input_proj(x)
        cls = self.cls_token.expand(batch_size, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.pos_encoder(h)

        cls_valid = torch.ones(batch_size, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_valid, mask], dim=1)  # True = valid
        padding_mask = ~full_mask  # Transformer expects True = pad

        token_embeddings = self.encoder(h, src_key_padding_mask=padding_mask)
        token_embeddings = self.norm(token_embeddings)

        cls_embedding = token_embeddings[:, 0, :]
        if self.config.pooling == "cls":
            chart_embedding = cls_embedding
        else:
            seq_tokens = token_embeddings[:, 1:, :]
            seq_valid = mask.unsqueeze(-1)
            denom = seq_valid.sum(dim=1).clamp_min(1)
            mean_embedding = (seq_tokens * seq_valid).sum(dim=1) / denom
            chart_embedding = torch.cat([cls_embedding, mean_embedding], dim=-1)

        logits = self.classifier(chart_embedding)
        return logits, chart_embedding, token_embeddings


def build_transformer_model(
    input_dim: int = 89,
    num_classes: int = 4,
    model_dim: int = 128,
    ff_dim: int = 256,
    num_heads: int = 4,
    num_layers: int = 3,
    dropout: float = 0.2,
    pooling: str = "cls",
) -> MVPTransformerClassifier:
    config = TransformerConfig(
        input_dim=input_dim,
        model_dim=model_dim,
        ff_dim=ff_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes,
        pooling=pooling,
    )
    return MVPTransformerClassifier(config)


def smoke_test() -> None:
    model = build_transformer_model()
    x = torch.randn(3, 20, 89)
    mask = torch.ones(3, 20, dtype=torch.bool)
    logits, chart_embedding, token_embeddings = model(x, mask)
    print("logits:", tuple(logits.shape))
    print("chart_embedding:", tuple(chart_embedding.shape))
    print("token_embeddings:", tuple(token_embeddings.shape))


if __name__ == "__main__":
    smoke_test()

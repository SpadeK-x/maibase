from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mvp_event_encoder import (
    EVENT_TRAIT_DIM,
    EVENT_TYPE_DIM,
    INNER_MASK_DIM,
    NUMERIC_BLOCK_END,
    NUMERIC_BLOCK_START,
    OUTER_SLOT_VOCAB_SIZE,
    SLIDE_SHAPE_GROUP_DIM,
)
from mvp_transformer_model import SinusoidalPositionalEncoding


EVENT_TYPE_SLICE = slice(0, EVENT_TYPE_DIM)
EVENT_TRAIT_SLICE = slice(EVENT_TYPE_SLICE.stop, EVENT_TYPE_SLICE.stop + EVENT_TRAIT_DIM)
SLIDE_SHAPE_SLICE = slice(EVENT_TRAIT_SLICE.stop, EVENT_TRAIT_SLICE.stop + SLIDE_SHAPE_GROUP_DIM)
OUTER_SLOT1_SLICE = slice(SLIDE_SHAPE_SLICE.stop, SLIDE_SHAPE_SLICE.stop + OUTER_SLOT_VOCAB_SIZE)
OUTER_SLOT2_SLICE = slice(OUTER_SLOT1_SLICE.stop, OUTER_SLOT1_SLICE.stop + OUTER_SLOT_VOCAB_SIZE)
INNER_MASK_SLICE = slice(NUMERIC_BLOCK_END, NUMERIC_BLOCK_END + INNER_MASK_DIM)


@dataclass
class MaskedEventModelConfig:
    input_dim: int = 84
    model_dim: int = 128
    ff_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.2
    max_len: int = 4096
    pooling: str = "cls"


class MaskedEventTransformer(nn.Module):
    def __init__(self, config: MaskedEventModelConfig) -> None:
        super().__init__()
        if config.pooling not in {"cls", "cls_mean"}:
            raise ValueError(f"Unsupported pooling mode: {config.pooling}")

        self.config = config
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.model_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.input_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.model_dim))
        self.pos_encoder = SinusoidalPositionalEncoding(config.model_dim, config.max_len + 1)

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

        self.event_type_head = nn.Linear(config.model_dim, EVENT_TYPE_DIM)
        self.event_trait_head = nn.Linear(config.model_dim, EVENT_TRAIT_DIM)
        self.slide_shape_head = nn.Linear(config.model_dim, SLIDE_SHAPE_GROUP_DIM)
        self.outer_slot1_head = nn.Linear(config.model_dim, OUTER_SLOT_VOCAB_SIZE)
        self.outer_slot2_head = nn.Linear(config.model_dim, OUTER_SLOT_VOCAB_SIZE)
        self.numeric_head = nn.Linear(config.model_dim, NUMERIC_BLOCK_END - NUMERIC_BLOCK_START)
        self.inner_mask_head = nn.Linear(config.model_dim, INNER_MASK_DIM)

    def encode(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        masked_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if masked_positions is not None:
            x = torch.where(masked_positions.unsqueeze(-1), self.mask_token.expand_as(x), x)

        h = self.input_proj(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.pos_encoder(h)

        cls_valid = torch.ones(x.size(0), 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_valid, valid_mask], dim=1)
        token_embeddings = self.encoder(h, src_key_padding_mask=~full_mask)
        token_embeddings = self.norm(token_embeddings)

        cls_embedding = token_embeddings[:, 0, :]
        seq_embeddings = token_embeddings[:, 1:, :]
        if self.config.pooling == "cls":
            chart_embedding = cls_embedding
        else:
            seq_valid = valid_mask.unsqueeze(-1)
            denom = seq_valid.sum(dim=1).clamp_min(1)
            mean_embedding = (seq_embeddings * seq_valid).sum(dim=1) / denom
            chart_embedding = torch.cat([cls_embedding, mean_embedding], dim=-1)
        return chart_embedding, seq_embeddings, token_embeddings

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        masked_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        chart_embedding, seq_embeddings, token_embeddings = self.encode(x, valid_mask, masked_positions=masked_positions)
        return {
            "chart_embedding": chart_embedding,
            "seq_embeddings": seq_embeddings,
            "token_embeddings": token_embeddings,
            "event_type_logits": self.event_type_head(seq_embeddings),
            "event_trait_logits": self.event_trait_head(seq_embeddings),
            "slide_shape_logits": self.slide_shape_head(seq_embeddings),
            "outer_slot1_logits": self.outer_slot1_head(seq_embeddings),
            "outer_slot2_logits": self.outer_slot2_head(seq_embeddings),
            "numeric_pred": self.numeric_head(seq_embeddings),
            "inner_mask_logits": self.inner_mask_head(seq_embeddings),
        }


def build_masked_event_model(
    input_dim: int = 84,
    model_dim: int = 128,
    ff_dim: int = 256,
    num_heads: int = 4,
    num_layers: int = 3,
    dropout: float = 0.2,
    pooling: str = "cls",
) -> MaskedEventTransformer:
    return MaskedEventTransformer(
        MaskedEventModelConfig(
            input_dim=input_dim,
            model_dim=model_dim,
            ff_dim=ff_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            pooling=pooling,
        )
    )


def build_masked_targets(batch_x: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "event_type": batch_x[:, :, EVENT_TYPE_SLICE].argmax(dim=-1),
        "event_trait": batch_x[:, :, EVENT_TRAIT_SLICE].argmax(dim=-1),
        "slide_shape": batch_x[:, :, SLIDE_SHAPE_SLICE].argmax(dim=-1),
        "outer_slot1": batch_x[:, :, OUTER_SLOT1_SLICE].argmax(dim=-1),
        "outer_slot2": batch_x[:, :, OUTER_SLOT2_SLICE].argmax(dim=-1),
        "numeric": batch_x[:, :, NUMERIC_BLOCK_START:NUMERIC_BLOCK_END],
        "inner_mask": batch_x[:, :, INNER_MASK_SLICE],
    }


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], targets[mask])


def masked_mse(pred: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return pred.sum() * 0.0
    diff = pred[mask] - targets[mask]
    return (diff * diff).mean()


def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[mask], targets[mask])


def compute_masked_modeling_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    masked_positions: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    losses = {
        "event_type": masked_cross_entropy(outputs["event_type_logits"], targets["event_type"], masked_positions),
        "event_trait": masked_cross_entropy(outputs["event_trait_logits"], targets["event_trait"], masked_positions),
        "slide_shape": masked_cross_entropy(outputs["slide_shape_logits"], targets["slide_shape"], masked_positions),
        "outer_slot1": masked_cross_entropy(outputs["outer_slot1_logits"], targets["outer_slot1"], masked_positions),
        "outer_slot2": masked_cross_entropy(outputs["outer_slot2_logits"], targets["outer_slot2"], masked_positions),
        "numeric": masked_mse(outputs["numeric_pred"], targets["numeric"], masked_positions),
        "inner_mask": masked_bce_with_logits(outputs["inner_mask_logits"], targets["inner_mask"], masked_positions),
    }
    total_loss = sum(losses.values())
    scalar_metrics = {name: float(value.detach().item()) for name, value in losses.items()}
    scalar_metrics["total"] = float(total_loss.detach().item())
    return total_loss, scalar_metrics


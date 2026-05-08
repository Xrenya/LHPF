from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..layers.mlp_layer import MLPLayer
from .planning_decoder import DecoderLayer


class LHPFDecoder(nn.Module):
    def __init__(
        self,
        num_mode,
        decoder_depth,
        dim,
        num_heads,
        mlp_ratio,
        dropout,
        future_steps,
    ) -> None:
        super().__init__()

        self.num_mode = num_mode
        self.future_steps = future_steps
        self.history_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        self.fusion_norm = nn.LayerNorm(dim)

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderLayer(dim, num_heads, mlp_ratio, dropout)
                for _ in range(decoder_depth)
            ]
        )

        self.loc_head = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.yaw_head = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.vel_head = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.pi_head = MLPLayer(dim, dim, 1)

    def forward(
        self,
        planning_embedding: Tensor,
        historical_planning_embedding: Tensor,
        enc_emb: Tensor,
        enc_key_padding_mask: Tensor,
        r_key_padding_mask: Tensor,
        m_pos: Tensor,
        historical_valid_mask: Optional[Tensor] = None,
    ):
        historical_embedding = self._aggregate_history(
            historical_planning_embedding,
            planning_embedding,
            historical_valid_mask,
        )
        q = planning_embedding + self.history_proj(historical_embedding)
        q = self.fusion_norm(q)

        for blk in self.decoder_blocks:
            q = blk(
                q,
                enc_emb,
                tgt_key_padding_mask=r_key_padding_mask,
                memory_key_padding_mask=enc_key_padding_mask,
                m_pos=m_pos,
            )
            assert torch.isfinite(q).all()

        traj, pi = self.decode(q)
        return traj, pi, q

    def decode(self, q: Tensor):
        bs, R, _, _ = q.shape

        loc = self.loc_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
        yaw = self.yaw_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
        vel = self.vel_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
        pi = self.pi_head(q).squeeze(-1)

        traj = torch.cat([loc, yaw, vel], dim=-1)
        return traj, pi

    def _aggregate_history(
        self,
        historical_planning_embedding: Tensor,
        planning_embedding: Tensor,
        historical_valid_mask: Optional[Tensor],
    ) -> Tensor:
        history = self._normalize_history_shape(
            historical_planning_embedding,
            planning_embedding,
        )

        if historical_valid_mask is None:
            return history.mean(dim=1)

        history_mask = self._normalize_history_mask(
            historical_valid_mask,
            history,
            planning_embedding,
        )
        history_mask = history_mask.unsqueeze(-1).unsqueeze(-1)
        history_mask = history_mask.to(dtype=history.dtype)
        history_sum = (history * history_mask).sum(dim=1)
        history_count = history_mask.sum(dim=1).clamp_min(1.0)
        return history_sum / history_count

    def _normalize_history_shape(
        self,
        historical_planning_embedding: Tensor,
        planning_embedding: Tensor,
    ) -> Tensor:
        bs, R, M, D = planning_embedding.shape
        history = historical_planning_embedding

        if history.dim() == 4:
            history = history.unsqueeze(1)
        elif history.dim() == 5:
            if history.shape[2] == R and history.shape[3] == M:
                pass
            elif history.shape[1] == R and history.shape[3] == M:
                history = history.permute(0, 2, 1, 3, 4)
        else:
            raise ValueError(
                "historical_planning_embedding must have shape "
                "(B, R, M, D), (B, H, R, M, D), or (B, R, H, M, D)"
            )

        if history.shape[0] != bs or history.shape[-1] != D:
            raise ValueError(
                "historical_planning_embedding batch size and hidden dimension "
                "must match planning_embedding"
            )

        if history.shape[2] == R and history.shape[3] == M:
            return history

        aligned = planning_embedding.new_zeros(
            bs,
            history.shape[1],
            R,
            M,
            D,
        )
        r_count = min(R, history.shape[2])
        m_count = min(M, history.shape[3])
        aligned[:, :, :r_count, :m_count] = history[:, :, :r_count, :m_count]
        return aligned

    def _normalize_history_mask(
        self,
        historical_valid_mask: Tensor,
        history: Tensor,
        planning_embedding: Tensor,
    ) -> Tensor:
        bs, H, R, _, _ = history.shape
        mask = historical_valid_mask.bool()

        if mask.dim() == 2:
            mask = mask[:, :, None].expand(bs, H, R)
        elif mask.dim() == 3:
            if mask.shape[1] == H and mask.shape[2] == R:
                pass
            elif mask.shape[1] == R and mask.shape[2] == H:
                mask = mask.permute(0, 2, 1)
        else:
            raise ValueError(
                "historical_planning_valid_mask must have shape "
                "(B, H), (B, H, R), or (B, R, H)"
            )

        if mask.shape[0] != bs:
            raise ValueError(
                "historical_planning_valid_mask batch size must match "
                "planning_embedding"
            )

        if mask.shape[1] == H and mask.shape[2] == R:
            return mask

        aligned = torch.zeros(
            bs,
            H,
            R,
            device=planning_embedding.device,
            dtype=torch.bool,
        )
        h_count = min(H, mask.shape[1])
        r_count = min(R, mask.shape[2])
        aligned[:, :h_count, :r_count] = mask[:, :h_count, :r_count]
        return aligned

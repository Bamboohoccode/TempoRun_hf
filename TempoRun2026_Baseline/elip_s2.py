"""ELIP-S-2 prompt mapper and SigLIP2 reranker.

Only ``prompt_mapper`` is trainable.  The SigLIP2 text/vision backbone remains
frozen.  A trained mapper checkpoint is mandatory: a random mapper is not an
ELIP model and will normally make retrieval worse.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers.masking_utils import create_bidirectional_mask
except ImportError as exc:  # pragma: no cover - depends on Transformers version
    raise ImportError(
        "ELIP-S-2 requires a recent Transformers release containing "
        "transformers.masking_utils.create_bidirectional_mask."
    ) from exc


class TextToVisualPrompts(nn.Module):
    def __init__(
        self,
        text_dim: int,
        vision_dim: int,
        num_prompts: int = 10,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_prompts = num_prompts
        self.vision_dim = vision_dim
        hidden_dim = hidden_dim or vision_dim

        self.mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_prompts * vision_dim),
        )

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        prompts = self.mlp(text_features)
        return prompts.reshape(
            text_features.shape[0], self.num_prompts, self.vision_dim
        )


class ELIPS2Reranker(nn.Module):
    """Query-conditioned image reranking for Hugging Face SigLIP2."""

    def __init__(
        self,
        clip_model,
        checkpoint: str | Path,
        num_prompts: int = 10,
        mapper_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.model = clip_model.model
        self.processor = clip_model.processor
        self.device = clip_model.device
        self.num_prompts = num_prompts

        text_config = self.model.config.text_config
        vision_config = self.model.config.vision_config
        text_dim = int(
            getattr(text_config, "projection_size", None)
            or text_config.hidden_size
        )
        vision_dim = int(vision_config.hidden_size)
        if text_dim != vision_dim:
            raise ValueError(
                f"SigLIP2 text dim ({text_dim}) and vision dim ({vision_dim}) "
                "must match for dot-product retrieval."
            )

        self.prompt_mapper = TextToVisualPrompts(
            text_dim=text_dim,
            vision_dim=vision_dim,
            num_prompts=num_prompts,
            hidden_dim=mapper_hidden_dim,
        ).to(self.device)

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        self.load_mapper(checkpoint)
        self.prompt_mapper.eval()

    def load_mapper(self, checkpoint: str | Path) -> None:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"ELIP mapper checkpoint not found: {checkpoint}")

        try:
            state = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
        except TypeError:  # Older PyTorch without weights_only.
            state = torch.load(checkpoint, map_location="cpu")

        if isinstance(state, dict) and "num_prompts" in state:
            saved_prompts = int(state["num_prompts"])
            if saved_prompts != self.num_prompts:
                raise ValueError(
                    f"Checkpoint uses {saved_prompts} prompts, but "
                    f"--num-prompts={self.num_prompts}."
                )

        if isinstance(state, dict) and "prompt_mapper" in state:
            state = state["prompt_mapper"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        if not isinstance(state, dict):
            raise TypeError("ELIP checkpoint does not contain a state dict")

        # Accept checkpoints saved from either mapper alone or the full module.
        cleaned = {}
        for key, value in state.items():
            key = key.removeprefix("module.")
            key = key.removeprefix("prompt_mapper.")
            cleaned[key] = value
        self.prompt_mapper.load_state_dict(cleaned, strict=True)

    def _prompted_image_features(
        self,
        pil_images: Sequence,
        query_features: torch.Tensor,
    ) -> torch.Tensor:
        inputs = self.processor(images=list(pil_images), return_tensors="pt")
        vision = self.model.vision_model
        vision_dtype = next(vision.parameters()).dtype

        pixel_values = inputs["pixel_values"].to(
            device=self.device, dtype=vision_dtype
        )
        spatial_shapes = inputs.get("spatial_shapes")
        if spatial_shapes is not None:
            spatial_shapes = spatial_shapes.to(self.device)

        try:
            patch_tokens = vision.embeddings(pixel_values, spatial_shapes)
        except TypeError:
            # Compatibility with a few older fixed-resolution implementations.
            patch_tokens = vision.embeddings(pixel_values)

        batch_size, num_patches = patch_tokens.shape[:2]
        pixel_mask = inputs.get("pixel_attention_mask")
        if pixel_mask is None:
            pixel_mask = torch.ones(
                (batch_size, num_patches),
                dtype=torch.long,
                device=self.device,
            )
        else:
            pixel_mask = pixel_mask.to(self.device)

        if query_features.ndim == 1:
            query_features = query_features.unsqueeze(0)
        query_features = F.normalize(
            query_features.to(self.device, dtype=torch.float32), dim=-1
        )
        if query_features.shape[0] == 1 and batch_size > 1:
            query_features = query_features.expand(batch_size, -1)
        if query_features.shape[0] != batch_size:
            raise ValueError("Need one query feature per image, or one shared query")

        mapper_dtype = next(self.prompt_mapper.parameters()).dtype
        prompts = self.prompt_mapper(
            query_features.to(dtype=mapper_dtype)
        ).to(dtype=patch_tokens.dtype)
        hidden_states = torch.cat([prompts, patch_tokens], dim=1)

        prompt_mask = torch.ones(
            (batch_size, self.num_prompts),
            dtype=pixel_mask.dtype,
            device=self.device,
        )
        full_mask = torch.cat([prompt_mask, pixel_mask], dim=1)
        encoder_mask = create_bidirectional_mask(
            config=vision.config,
            inputs_embeds=hidden_states,
            attention_mask=full_mask,
        )

        outputs = vision.encoder(
            inputs_embeds=hidden_states,
            attention_mask=encoder_mask,
        )
        hidden_states = vision.post_layernorm(outputs.last_hidden_state)

        # Prompts influence patch tokens through self-attention.  Pool only the
        # image tokens, matching the original image-representation semantics.
        patch_states = hidden_states[:, self.num_prompts :]
        image_features = vision.head(patch_states, pixel_mask)
        return F.normalize(image_features.float(), dim=-1)

    @torch.inference_mode()
    def rerank(
        self,
        query_feature: np.ndarray | torch.Tensor,
        pil_images: Sequence,
        batch_size: int = 16,
    ) -> np.ndarray:
        if isinstance(query_feature, np.ndarray):
            query_feature = torch.from_numpy(query_feature)
        query_feature = F.normalize(
            query_feature.to(self.device, dtype=torch.float32).reshape(1, -1),
            dim=-1,
        )

        scores: list[torch.Tensor] = []
        for start in range(0, len(pil_images), batch_size):
            images = pil_images[start : start + batch_size]
            image_features = self._prompted_image_features(
                images, query_feature
            )
            scores.append((image_features @ query_feature.T)[:, 0].cpu())

        if not scores:
            return np.empty(0, dtype=np.float32)
        return torch.cat(scores).numpy()

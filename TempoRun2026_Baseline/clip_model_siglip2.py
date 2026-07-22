"""SigLIP2 wrapper for image/text embedding extraction.

The returned NumPy arrays are always L2-normalized, so a dot product is a
cosine-similarity score.  This wrapper also handles recent Transformers
versions where ``get_*_features`` returns ``BaseModelOutputWithPooling``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor


def _pooler_output(output: object) -> torch.Tensor:
    """Return the feature tensor across old and new Transformers APIs."""
    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if pooled is None:
        raise TypeError(
            "SigLIP2 feature output has no pooler_output; "
            "check the installed Transformers version."
        )
    return pooled


class ClipModel(nn.Module):
    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        precision: str = "fp16",
        device: str | None = None,
        freeze_vision: bool = True,
        max_text_length: int = 64,
    ) -> None:
        super().__init__()

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_name = model_name
        self.max_text_length = max_text_length

        if precision == "fp16" and self.device.type == "cuda":
            self.model_dtype = torch.float16
            self.np_dtype = np.float16
        elif precision == "bf16" and self.device.type == "cuda":
            self.model_dtype = torch.bfloat16
            # NumPy has no portable bfloat16 representation.
            self.np_dtype = np.float32
        else:
            self.model_dtype = torch.float32
            self.np_dtype = np.float32

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=self.model_dtype,
        ).to(self.device)
        self.model.eval()

        if freeze_vision:
            for parameter in self.model.vision_model.parameters():
                parameter.requires_grad_(False)

        text_config = self.model.config.text_config
        self.dim = int(
            getattr(text_config, "projection_size", None)
            or text_config.hidden_size
        )

    @torch.inference_mode()
    def encode_images(
        self,
        pil_images: Sequence,
        batch_size: int = 64,
    ) -> np.ndarray:
        features: list[np.ndarray] = []

        for start in range(0, len(pil_images), batch_size):
            batch = list(pil_images[start : start + batch_size])
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = inputs.to(self.device)

            output = self.model.get_image_features(
                pixel_values=inputs["pixel_values"],
                pixel_attention_mask=inputs.get("pixel_attention_mask"),
                spatial_shapes=inputs.get("spatial_shapes"),
            )
            tensor = F.normalize(_pooler_output(output).float(), dim=-1)
            features.append(tensor.cpu().numpy().astype(self.np_dtype))

        if features:
            return np.concatenate(features, axis=0)
        return np.zeros((0, self.dim), dtype=self.np_dtype)

    @torch.inference_mode()
    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 256,
    ) -> np.ndarray:
        features: list[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            inputs = self.processor(
                text=batch,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            ).to(self.device)

            output = self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            tensor = F.normalize(_pooler_output(output).float(), dim=-1)
            features.append(tensor.cpu().numpy().astype(self.np_dtype))

        if features:
            return np.concatenate(features, axis=0)
        return np.zeros((0, self.dim), dtype=self.np_dtype)

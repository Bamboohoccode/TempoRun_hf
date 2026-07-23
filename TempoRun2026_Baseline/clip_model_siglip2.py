"""OpenCLIP wrapper for ViT-gopt-16-SigLIP2-384 Stage-1 retrieval."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipModel(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-gopt-16-SigLIP2-384",
        pretrained: str | None = "webli",
        precision: str = "fp16",
        device: str | None = None,
        freeze_vision: bool = True,
        max_text_length: int = 64,
    ) -> None:
        super().__init__()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cpu" and precision != "fp32":
            precision = "fp32"

        pretrained = None if pretrained in (None, "", "None") else pretrained
        self.model, self.preprocess_train, self.preprocess_val = (
            open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                precision=precision,
                device=self.device,
            )
        )
        self.model.eval()
        try:
            self.tokenizer = open_clip.get_tokenizer(
                model_name, context_length=max_text_length
            )
        except TypeError:
            self.tokenizer = open_clip.get_tokenizer(model_name)

        if freeze_vision:
            for parameter in self.model.visual.parameters():
                parameter.requires_grad_(False)

        self.model_dtype = next(self.model.visual.parameters()).dtype
        self.np_dtype = (
            np.float16 if self.model_dtype == torch.float16 else np.float32
        )
        self.dim = int(
            getattr(self.model.visual, "output_dim", 0)
            or getattr(getattr(self.model, "text", None), "output_dim", 0)
            or 1536
        )

    @torch.inference_mode()
    def encode_images(
        self, pil_images: Sequence, batch_size: int = 4
    ) -> np.ndarray:
        features: list[np.ndarray] = []
        for start in range(0, len(pil_images), batch_size):
            batch = [
                self.preprocess_val(image)
                for image in pil_images[start : start + batch_size]
            ]
            if not batch:
                continue
            pixels = torch.stack(batch).to(
                self.device, dtype=self.model_dtype, non_blocking=True
            )
            output = self.model.encode_image(pixels)
            output = F.normalize(output.float(), dim=-1)
            features.append(output.cpu().numpy().astype(self.np_dtype))
        if features:
            return np.concatenate(features, axis=0)
        return np.zeros((0, self.dim), dtype=self.np_dtype)

    @torch.inference_mode()
    def encode_texts(
        self, texts: Sequence[str], batch_size: int = 32
    ) -> np.ndarray:
        features: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            if not batch:
                continue
            tokens = self.tokenizer(batch).to(self.device, non_blocking=True)
            output = self.model.encode_text(tokens)
            output = F.normalize(output.float(), dim=-1)
            features.append(output.cpu().numpy().astype(self.np_dtype))
        if features:
            return np.concatenate(features, axis=0)
        return np.zeros((0, self.dim), dtype=self.np_dtype)

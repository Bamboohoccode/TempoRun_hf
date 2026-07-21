"""Self-contained CLIP wrapper (open_clip) — image AND text encoding, L2-normalized."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from transformers import AutoModel, AutoProcessor
class ClipModel(nn.Module):
    def __init__(
        self,
        model_name="google/siglip2-base-patch16-224",
        precision="fp16",
        device=None,
        freeze_vision=True,
        max_text_length=64,
    ):
        super().__init__()

        self.device = torch.device(
            device or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.processor = AutoProcessor.from_pretrained(model_name)
        if precision == "fp16" and self.device.type == "cuda":
            self.model_dtype = torch.float16
            self.np_dtype = np.float16
        elif precision == "bf16" and self.device.type == "cuda":
            self.model_dtype = torch.bfloat16
            self.np_dtype = np.float32
        else:
            self.model_dtype = torch.float32
            self.np_dtype = np.float32
        
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=self.model_dtype).to(self.device)
        self.model.eval()
        self.max_text_length = max_text_length 
        if freeze_vision:
            for parameter in self.model.vision_model.parameters():
                parameter.requires_grad_(False)

        self.dim = (
        self.model.config.text_config.projection_size
        or self.model.config.text_config.hidden_size)
        self.to(self.device)
        
    def encode_images(self, pil_images: list, batch_size=64) -> np.ndarray:
        feats = []
        for i in range(0, len(pil_images), batch_size):
            img_batch = pil_images[i:i + batch_size]
            if len(img_batch) == 0:
                continue
            batch = self.processor(images = img_batch, return_tensors = "pt")
            img_inputs = {
                key: value.to(self.device) for key,value in batch.items() if key in {"pixel_values","pixel_attention_mask","spatial_shapes"}
            }
            img_inputs["pixel_values"] = img_inputs[
                "pixel_values"
            ].to(dtype=self.model_dtype)# Only pixel_values need to transform in fp16/fp32

            with torch.inference_mode():
                features = self.model.get_image_features(**img_inputs)
                if not isinstance(features, torch.Tensor):
                    features = features.pooler_output
                features = F.normalize(features,dim = -1)
                features = (features.float().cpu().numpy().astype(self.np_dtype))
                feats.append(features)
        return np.concatenate(feats,axis =0) if feats else np.zeros((0, self.dim),self.np_dtype)

    def encode_texts(self, texts: list[str], batch_size=256) -> np.ndarray:
        feats = []
        for i in range(0, len(texts), batch_size):

            text_batch = texts[i:i + batch_size]
            if len(text_batch) == 0:
                continue
            toks = self.processor(text = text_batch,
                                  padding = "max_length",
                                  truncation = True,
                                  max_length = self.max_text_length,
                                  return_tensors = "pt").to(self.device)
            with torch.inference_mode():
                features = self.model.get_text_features(**toks)
                if not isinstance(features, torch.Tensor):
                    features = features.pooler_output
                features = F.normalize(features,dim = -1)
                feats.append(features.float().cpu().numpy().astype(self.np_dtype))
    
        return np.concatenate(feats, 0) if feats else np.zeros((0, self.dim), (self.np_dtype))
    

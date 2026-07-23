"""Official ELIP-S-2 reranker for ViT-gopt-16-SigLIP2-384.

The ELIP repository must be cloned and its patched timm vision_transformer.py
installed before this module is imported. Stage-1-only retrieval does not need
this module or an ELIP checkpoint.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"ELIP checkpoint has no state dict: {path}")
    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
    }


class ELIPS2Reranker:
    """Load the official query-conditioned ELIP-S-2 vision model."""

    def __init__(
        self,
        checkpoint: str | Path,
        elip_repo: str | Path,
        model_name: str = "ViT-gopt-16-SigLIP2-384",
        pretrained: str = "webli",
        precision: str = "fp16",
        device: str = "cuda:0",
        max_text_length: int = 64,
    ) -> None:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"ELIP checkpoint not found: {checkpoint}")

        source_dir = Path(elip_repo) / "ELIP-C" / "src"
        if not (source_dir / "open_clip" / "factory.py").is_file():
            raise FileNotFoundError(
                f"Official ELIP source not found below: {source_dir}"
            )
        sys.path.insert(0, str(source_dir))

        # This must resolve to ELIP-C/src/open_clip, not pip open_clip.
        import open_clip
        from open_clip import factory as elip_factory

        module_path = Path(open_clip.__file__).resolve()
        if source_dir.resolve() not in module_path.parents:
            raise RuntimeError(
                "pip open_clip was imported before official ELIP open_clip. "
                "Run reranking in a fresh Python process."
            )

        self.device = torch.device(device)
        if self.device.type == "cpu" and precision != "fp32":
            precision = "fp32"

        # The public HF config supplies the model config missing from the ELIP
        # repository. Download only the small config, not the 7.5 GB base
        # checkpoint: the official ELIP .pt already contains the full model.
        from huggingface_hub import hf_hub_download

        model_id = f"timm/{model_name}"
        config_path = hf_hub_download(
            repo_id=model_id, filename="open_clip_config.json"
        )
        with open(config_path, encoding="utf-8") as stream:
            hub_config = json.load(stream)
        elip_factory._MODEL_CONFIGS[model_name] = hub_config["model_cfg"]
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=None,
            precision=precision,
            device=self.device,
            force_preprocess_cfg=hub_config["preprocess_cfg"],
        )
        try:
            self.tokenizer = open_clip.get_tokenizer(
                model_name, context_length=max_text_length
            )
        except TypeError:
            self.tokenizer = open_clip.get_tokenizer(model_name)

        state = _checkpoint_state(checkpoint)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.model_dtype = next(self.model.visual.parameters()).dtype
        self.np_dtype = (
            np.float16 if self.model_dtype == torch.float16 else np.float32
        )
        self.dim = int(
            getattr(self.model.visual, "output_dim", 0)
            or getattr(getattr(self.model, "text", None), "output_dim", 0)
            or 1536
        )
        self.pretrained = pretrained

    @torch.inference_mode()
    def encode_texts(
        self, texts: Sequence[str], batch_size: int = 32
    ) -> np.ndarray:
        features: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            tokens = self.tokenizer(batch).to(self.device, non_blocking=True)
            output = self.model.encode_text(tokens, normalize=True)
            features.append(output.float().cpu().numpy().astype(self.np_dtype))
        if features:
            return np.concatenate(features, axis=0)
        return np.zeros((0, self.dim), dtype=self.np_dtype)

    @torch.inference_mode()
    def rerank(
        self,
        query_feature: np.ndarray | torch.Tensor,
        pil_images: Sequence,
        batch_size: int = 2,
    ) -> np.ndarray:
        query = torch.as_tensor(
            query_feature, device=self.device, dtype=torch.float32
        ).reshape(1, -1)
        query = F.normalize(query, dim=-1)
        scores: list[torch.Tensor] = []

        for start in range(0, len(pil_images), batch_size):
            images = pil_images[start : start + batch_size]
            pixels = torch.stack(
                [self.preprocess(image) for image in images]
            ).to(self.device, dtype=self.model_dtype, non_blocking=True)
            text_features = query.expand(len(images), -1).to(self.model_dtype)
            # Positional argument supports the official CustomTextCLIP API.
            image_features = self.model.encode_image(
                pixels, text_features, normalize=True
            )
            scores.append((image_features.float() @ query.T)[:, 0].cpu())

        if not scores:
            return np.empty(0, dtype=np.float32)
        return torch.cat(scores).numpy()

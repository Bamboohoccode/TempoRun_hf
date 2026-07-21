import torch
import torch.nn as nn
import numpy as np
from functools import partial
from clip_model import ClipModel
class LoRALayer(nn.Module):
    def __init__(self, layer, rank, alpha):
        super().__init__()

        device = layer.weight.device

        # Với rank=8, alpha=16 thì scale=2
        self.scale = alpha / rank

        # Tham số trainable giữ fp32
        self.A = nn.Parameter(
            torch.empty(
                layer.in_features,
                rank,
                device=device,
                dtype=torch.float32,
            )
        )

        self.B = nn.Parameter(
            torch.zeros(
                rank,
                layer.out_features,
                device=device,
                dtype=torch.float32,
            )
        )

        nn.init.normal_(
            self.A,
            mean=0.0,
            std=rank ** -0.5,
        )

    def forward(self, x):
        # Autocast bên ngoài có thể tự đưa matmul về fp16,
        # nên cần tắt autocast cho riêng nhánh LoRA.
        with torch.autocast(
            device_type=x.device.type,
            enabled=False,
        ):
            delta = (
                x.float() @ self.A @ self.B
            ) * self.scale

        return delta


class LayerWLoRA(nn.Module):
    def __init__(self, layer, rank, alpha):
        super().__init__()

        self.layer = layer
        self.LoRA = LoRALayer(
            layer=layer,
            rank=rank,
            alpha=alpha,
        )

    def forward(self, x):
        base_output = self.layer(x)       # fp16
        lora_output = self.LoRA(x)        # fp32

        # Đổi LoRA output sang dtype của nhánh gốc
        return base_output + lora_output.to(
            dtype=base_output.dtype)
#CONFIG
default_r = 8
default_alpha = 16
add_LoRA = partial(LayerWLoRA,rank = default_r,alpha = default_alpha)
def assign_LoRA(model,    
    lora_r = 8,
    lora_alpha = 16,
    LoRA_attn_qkv = True,
    LoRA_attn_proj = True,
    LoRA_attn_mlp = False,
    num_blocks_visual = 4,
    LoRA_attnP_qkv = False,
    LoRA_attnP_proj = False,
    LoRA_attnP_mlp = False,
    LoRA_head = True,
    LoRA_text_mlp = False):

    num_blocks = len(model.model.visual.trunk.blocks)
    #For image
    for i,layer in enumerate(model.model.visual.trunk.blocks):
        if(i < num_blocks - num_blocks_visual): continue
        print(f"Block thứ {i} được gắn LoRA")
        if(LoRA_attn_qkv): layer.attn.qkv = add_LoRA(layer.attn.qkv,rank = lora_r,alpha = lora_alpha)
        if(LoRA_attn_proj): layer.attn.proj = add_LoRA(layer.attn.proj,rank = lora_r,alpha = lora_alpha)
        if(LoRA_attn_mlp):
            layer.mlp.fc1 = add_LoRA(layer.mlp.fc1,rank = lora_r,alpha = lora_alpha)
            layer.mlp.fc2 = add_LoRA(layer.mlp.fc2,rank = lora_r,alpha = lora_alpha)
    if(LoRA_attnP_qkv):
        model.model.visual.trunk.attn_pool.q = add_LoRA(model.model.visual.trunk.attn_pool.q,rank = lora_r,alpha = lora_alpha)
        model.model.visual.trunk.attn_pool.kv = add_LoRA(model.model.visual.trunk.attn_pool.kv,rank = lora_r,alpha = lora_alpha)
    if(LoRA_attnP_proj):
        model.model.visual.trunk.attn_pool.proj = add_LoRA(model.model.visual.trunk.attn_pool.proj,rank = lora_r,alpha = lora_alpha)
    if LoRA_attnP_mlp:
        model.model.visual.trunk.attn_pool.mlp.fc1 = add_LoRA(model.model.visual.trunk.attn_pool.mlp.fc1,rank = lora_r,alpha = lora_alpha)
        model.model.visual.trunk.attn_pool.mlp.fc2 = add_LoRA(model.model.visual.trunk.attn_pool.mlp.fc2,rank = lora_r,alpha = lora_alpha)
    if(LoRA_head):
        model.model.visual.trunk.head = add_LoRA(model.model.visual.trunk.head,rank = lora_r,alpha = lora_alpha)
    #For text
    for layer in model.model.text.transformer.resblocks:
        if(LoRA_text_mlp):
            layer.mlp.c_fc = add_LoRA(layer.mlp.c_fc,rank = lora_r,alpha = lora_alpha)
            layer.mlp.c_proj = add_LoRA(layer.mlp.c_proj,rank = lora_r,alpha = lora_alpha)

def Apply_weights(model,device,pth_dir):
    print("Loading weights ....")

    checkpoint = torch.load(
        pth_dir,
        map_location=device,
        weights_only=True,)
    model.load_state_dict(checkpoint)
    model = model.float().to(device)
    print(f"Data type cua model la: {next(model.parameters()).dtype}")
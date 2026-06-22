from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def resolve_decoder_layers(model: nn.Module):
    """Return decoder layers for common LLaMA/Qwen-style causal LMs."""
    current = model
    while current is not None:
        inner = getattr(current, "model", None)
        if inner is not None:
            if hasattr(inner, "layers"):
                return inner.layers
            decoder = getattr(inner, "decoder", None)
            if decoder is not None and hasattr(decoder, "layers"):
                return decoder.layers

        if hasattr(current, "layers"):
            return current.layers
        decoder = getattr(current, "decoder", None)
        if decoder is not None and hasattr(decoder, "layers"):
            return decoder.layers

        transformer = getattr(current, "transformer", None)
        if transformer is not None and hasattr(transformer, "h"):
            return transformer.h

        current = getattr(current, "base_model", None)
    raise AttributeError("Unable to locate decoder layers on the provided model.")


def _clone_linear_out(linear: nn.Linear, indices: torch.Tensor) -> nn.Linear:
    if indices.dim() != 1:
        raise ValueError("indices must be 1D.")
    device = linear.weight.device
    dtype = linear.weight.dtype
    indices = indices.to(device=device, dtype=torch.long)
    new_linear = nn.Linear(
        linear.in_features,
        indices.numel(),
        bias=linear.bias is not None,
        device=device,
        dtype=dtype,
    )
    new_linear.weight.data.copy_(linear.weight.data.index_select(0, indices))
    if linear.bias is not None:
        new_linear.bias.data.copy_(linear.bias.data.index_select(0, indices))
    new_linear.weight.requires_grad_(linear.weight.requires_grad)
    if new_linear.bias is not None:
        new_linear.bias.requires_grad_(linear.bias.requires_grad)
    return new_linear


def _clone_linear_in(linear: nn.Linear, indices: torch.Tensor) -> nn.Linear:
    if indices.dim() != 1:
        raise ValueError("indices must be 1D.")
    device = linear.weight.device
    dtype = linear.weight.dtype
    indices = indices.to(device=device, dtype=torch.long)
    new_linear = nn.Linear(
        indices.numel(),
        linear.out_features,
        bias=linear.bias is not None,
        device=device,
        dtype=dtype,
    )
    new_linear.weight.data.copy_(linear.weight.data.index_select(1, indices))
    if linear.bias is not None:
        new_linear.bias.data.copy_(linear.bias.data)
    new_linear.weight.requires_grad_(linear.weight.requires_grad)
    if new_linear.bias is not None:
        new_linear.bias.requires_grad_(linear.bias.requires_grad)
    return new_linear


def load_topk_indices(path: str | Path) -> torch.Tensor:
    """Load per-layer FFN keep indices from a .pt/.pth or JSON file."""
    path = Path(path)
    if path.suffix.lower() in {".pt", ".pth"}:
        value = torch.load(path, map_location="cpu")
        if isinstance(value, dict):
            for key in ("topk_indices", "indices", "keep_indices"):
                if key in value:
                    value = value[key]
                    break
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        return value.long()

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        for key in ("topk_indices", "indices", "keep_indices"):
            if key in payload:
                payload = payload[key]
                break
    return torch.as_tensor(payload, dtype=torch.long)


def prune_llama_ffn(
    model: nn.Module,
    topk_indices: torch.Tensor,
    *,
    new_intermediate_size: Optional[int] = None,
) -> int:
    """Structurally prune LLaMA/Qwen-style FFN layers using per-layer indices.

    The function keeps the selected FFN intermediate channels in each decoder
    layer and rewrites `up_proj`, `gate_proj`, and `down_proj` accordingly.
    """
    if topk_indices.dim() == 3:
        topk_indices = topk_indices[0]
    if topk_indices.dim() != 2:
        raise ValueError(f"topk_indices must be 2D or 3D, got shape {tuple(topk_indices.shape)}.")

    layers = resolve_decoder_layers(model)
    if len(layers) != topk_indices.size(0):
        raise ValueError(
            f"Layer count mismatch: model has {len(layers)} layers, "
            f"indices have {topk_indices.size(0)}."
        )

    target_k = int(topk_indices.size(1))
    if new_intermediate_size is not None and int(new_intermediate_size) != target_k:
        raise ValueError(
            f"new_intermediate_size ({new_intermediate_size}) must match "
            f"indices length ({target_k})."
        )

    for layer_idx, layer in enumerate(layers):
        mlp = layer.mlp
        indices = topk_indices[layer_idx].to(device=mlp.up_proj.weight.device, dtype=torch.long)
        indices, _ = torch.sort(indices)

        mlp.up_proj = _clone_linear_out(mlp.up_proj, indices)
        gate_proj = getattr(mlp, "gate_proj", None)
        if gate_proj is not None:
            mlp.gate_proj = _clone_linear_out(gate_proj, indices)
        mlp.down_proj = _clone_linear_in(mlp.down_proj, indices)

        if hasattr(mlp, "intermediate_size"):
            mlp.intermediate_size = target_k

    if hasattr(model, "config") and model.config is not None:
        model.config.intermediate_size = target_k
    return target_k


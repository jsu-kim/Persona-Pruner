from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    normalized = dtype.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_tokenizer(model_name_or_path: str, *, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer


def load_causal_lm(
    model_name_or_path: str,
    *,
    dtype: str | torch.dtype = "bf16",
    device_map: Optional[str] = None,
    trust_remote_code: bool = False,
    attn_implementation: Optional[str] = None,
):
    kwargs = {
        "torch_dtype": resolve_dtype(dtype),
        "trust_remote_code": trust_remote_code,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    if device_map is None and torch.cuda.is_available():
        model.to("cuda")
    model.eval()
    return model


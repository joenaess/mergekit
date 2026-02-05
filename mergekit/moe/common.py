# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import logging
from typing import Dict, Optional, Tuple

import torch
import tqdm
import transformers
import math

from mergekit.architecture import WeightInfo
from mergekit.common import ModelReference, dtype_from_name
from mergekit.io import LazyTensorLoader, TensorWriter
from mergekit.merge import MergeOptions
from mergekit.moe.config import Expert, MoEMergeConfig


def initialize_io(
    config: MoEMergeConfig,
    out_path: str,
    merge_options: MergeOptions,
) -> Tuple[Dict[ModelReference, LazyTensorLoader], LazyTensorLoader, TensorWriter]:
    base_model = config.base_model
    loaders: Dict[ModelReference, LazyTensorLoader] = {}
    for model in tqdm.tqdm(
        [base_model] + [e.source_model for e in config.experts], desc="Warm up loaders"
    ):
        loaders[model] = model.lazy_loader(
            cache_dir=merge_options.transformers_cache,
            lazy_unpickle=merge_options.lazy_unpickle,
        )

    base_loader = loaders.get(base_model)
    writer = TensorWriter(
        out_path=out_path,
        max_shard_size=merge_options.out_shard_size,
        safe_serialization=merge_options.safe_serialization,
        use_async=merge_options.async_write,
        max_write_threads=merge_options.write_threads,
    )

    return loaders, base_loader, writer


def select_dtype(
    config: MoEMergeConfig, base_cfg: transformers.PretrainedConfig
) -> Optional[torch.dtype]:
    out_dtype = None
    if config.dtype:
        out_dtype = dtype_from_name(config.dtype)

    if out_dtype is None and base_cfg.torch_dtype:
        out_dtype = base_cfg.torch_dtype
        if isinstance(out_dtype, str):
            out_dtype = dtype_from_name(out_dtype)
    return out_dtype


def noise_and_scale(
    tensor: torch.Tensor, expert: Expert, is_residual: bool = False
) -> torch.Tensor:
    if expert.noise_scale is not None:
        noise = torch.randn_like(tensor) * expert.noise_scale
        tensor = tensor + noise
    if is_residual and expert.residual_scale is not None:
        tensor = tensor * expert.residual_scale
    return tensor


def copy_tensor_out(
    weight_info: WeightInfo,
    loader: LazyTensorLoader,
    writer: TensorWriter,
    expert: Optional[Expert] = None,
    is_residual: bool = False,
    output_name: Optional[str] = None,
    out_dtype: Optional[torch.dtype] = None,
    clone: bool = False,
):
    out_tensor_name = output_name or weight_info.name
    aliases = weight_info.aliases or []
    if not weight_info.optional:
        aliases += weight_info.tied_names or []
    try:
        tensor = loader.get_tensor(
            weight_info.name,
            aliases=aliases,
        )
    except KeyError:
        tensor = None
    if tensor is None:
        if weight_info.optional:
            return
        logging.error(f"Missing weight: {weight_info.name} / {out_tensor_name}")
        raise KeyError(out_tensor_name)

    if expert:
        tensor = noise_and_scale(tensor, expert, is_residual=is_residual)
    writer.save_tensor(
        out_tensor_name,
        tensor.to(dtype=out_dtype),
        clone=clone,
    )

def fuse_moe_ct_weights(
    base: torch.Tensor, 
    expert: torch.Tensor, 
    alpha: float
) -> torch.Tensor:
    """
    Residual-Expert Fusion (MoE-CT).
    Combines the original dense knowledge (Stability) with new experts (Plasticity).
    """
    if base.shape != expert.shape:
        raise ValueError(f"Shape mismatch: Base {base.shape} vs Expert {expert.shape}")
    
    # lerp: out = start + weight * (end - start)
    # We compute in float32 for high-precision blending, then cast back
    res = torch.lerp(base.to(torch.float32), expert.to(torch.float32), alpha)
    return res.to(base.dtype)


def get_moe_ct_alpha(
    layer_idx: int, 
    total_layers: int, 
    base_alpha: float, 
    strategy: str = "constant"
) -> float:
    denom = max(1, total_layers - 1)
    depth = layer_idx / denom
    
    if strategy == "constant":
        return base_alpha
    elif strategy == "linear_increase":
        return base_alpha * depth
    elif strategy == "linear_decrease":
        return base_alpha * (1.0 - depth)
    
    # NEW: U-Shaped Strategy
    elif strategy == "u_shaped":
        # Uses a cosine-based curve to dip in the middle
        # Alpha is high at depth 0.0 and 1.0, and low at 0.5
        curve = 0.5 * (1 + math.cos(2 * math.pi * depth))
        return base_alpha * curve
    
    return base_alpha


def fuse_gate_weights(
    base_gate: torch.Tensor, 
    expert_gates: torch.Tensor, 
    base_weight: float = 1.0
) -> torch.Tensor:
    """
    Adjusts router logits to favor the base model's knowledge.
    
    base_weight > 1.0: Router is biased towards 'Stability' (original FFN).
    base_weight < 1.0: Router is biased towards 'Plasticity' (new Experts).
    """
    # In MoE architectures like Qwen, the gate is a Linear layer (num_experts, hidden_size)
    # apply the weight to the rows corresponding to the 'shared' or 'anchor' experts
    modified_gates = expert_gates.clone()
    # If using a shared expert (index 0 or separate), scale its selection probability
    modified_gates *= base_weight
    return modified_gates
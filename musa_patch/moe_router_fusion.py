"""Bind TE fused MoE router APIs into Megatron even when TE version < 2.7.

Megatron-LM v0.16 only imports fused_topk_with_score_function if
is_te_min_version("2.7.0.dev"). This MUSA TE 2.0.0 already ships the kernel
in transformer_engine.pytorch.router.
"""
from transformer_engine.pytorch.router import (
    fused_compute_score_for_moe_aux_loss,
    fused_moe_aux_loss,
    fused_topk_with_score_function,
)
import megatron.core.extensions.transformer_engine as te_ext
import megatron.core.transformer.moe.moe_utils as moe_utils

te_ext.fused_topk_with_score_function = fused_topk_with_score_function
te_ext.fused_compute_score_for_moe_aux_loss = fused_compute_score_for_moe_aux_loss
te_ext.fused_moe_aux_loss = fused_moe_aux_loss
moe_utils.fused_topk_with_score_function = fused_topk_with_score_function
moe_utils.fused_compute_score_for_moe_aux_loss = fused_compute_score_for_moe_aux_loss
moe_utils.fused_moe_aux_loss = fused_moe_aux_loss

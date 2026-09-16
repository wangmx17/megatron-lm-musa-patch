# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain the MiniCPM5 16A3B GPT-MoE model on Megatron v0.19."""

from contextlib import nullcontext
import inspect
import os

import torch
# if os.getenv("ACCELERATOR_BACKEND", "musa") == "musa":
if os.getenv("ACCELERATOR_BACKEND") == "musa":
    import musa_patch
else:
    import cuda_patch
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, pretrain, print_rank_0
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from v019_pretrain import forward_step, train_valid_test_datasets_provider


def model_provider(
    pre_process=True,
    post_process=True,
    vp_stage=None,
    config=None,
    pg_collection=None,
) -> GPTModel:
    """Builds the model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        GPTModel: The Megatron Core GPT model.
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,

            # record stack information for the trace events
            trace_alloc_record_context=True)

    print_rank_0('building GPT model ...')
    if config is None:
        # Experimental loading arguments from yaml
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        if args.num_experts:
            # Define the decoder block spec
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config,
                use_transformer_engine=use_te,
                normalization=args.normalization,
                qk_l2_norm=args.qk_l2_norm,
                vp_stage=vp_stage,
            )
        else:
            # Define the decoder layer spec
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                    num_experts=args.num_experts,
                    moe_grouped_gemm=args.moe_grouped_gemm,
                    qk_layernorm=args.qk_layernorm,
                    multi_latent_attention=args.multi_latent_attention,
                    qk_l2_norm=args.qk_l2_norm,
                )
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(
                    num_experts=args.num_experts,
                    moe_grouped_gemm=args.moe_grouped_gemm,
                    qk_layernorm=args.qk_layernorm,
                    multi_latent_attention=args.multi_latent_attention,
                    normalization=args.normalization,
                    qk_l2_norm=args.qk_l2_norm,
                )

    build_model_context = nullcontext
    build_model_context_args = {}
    if args.fp8_param_gather:
        try:
            from transformer_engine.pytorch import fp8_model_init

            build_model_context = fp8_model_init
            build_model_context_args["enabled"] = True

            # Check if fp8_model_init supports preserve_high_precision_init_val
            if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                build_model_context_args["preserve_high_precision_init_val"] = True
        except ImportError as exc:
            raise RuntimeError(
                "--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found."
            ) from exc

    with build_model_context(**build_model_context_args):
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )

    return model


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    args = parse_and_validate_args(
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
    full_config = pretrain_cfg_container_from_args(args)
    if os.getenv('V019_VALIDATE_ONLY') == '1':
        print('V019_VALIDATE_ONLY: configuration constructed; training not launched', flush=True)
        raise SystemExit(0)
    pretrain(
        full_config,
        train_valid_test_datasets_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        model_provider=model_provider,
    )
    # Keep all ranks alive until every node has completed the last optimizer
    # step and logger flush.  Without this, the first torchrun to exit can tear
    # down MCCL while a peer is still leaving Megatron's training loop.
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.musa.synchronize()

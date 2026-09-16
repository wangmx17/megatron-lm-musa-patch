"""MUSA-only Megatron argument and transformer-config extensions.

Megatron Core 0.19 generates most command-line options from its config
dataclasses. Replacing ``_add_moe_args`` with the old 0.14 implementation would
remove those generated options and their newer semantics. The wrappers below
always call the 0.19 implementation first and add only options that are still
private to the MUSA stack.
"""

from __future__ import annotations

import os
from argparse import ArgumentParser, _ArgumentGroup
from typing import Any

import megatron.training.arguments as training_arguments
from megatron.training import argument_utils


def _add_argument_if_missing(
    parser: ArgumentParser,
    group: _ArgumentGroup,
    *option_strings: str,
    **kwargs: Any,
) -> None:
    """Add an option without colliding with a newer Megatron definition."""

    if any(option in parser._option_string_actions for option in option_strings):
        return
    group.add_argument(*option_strings, **kwargs)


_ORIGINAL_ADD_MOE_ARGS = getattr(
    training_arguments,
    "_musa_original_add_moe_args",
    training_arguments._add_moe_args,
)
training_arguments._musa_original_add_moe_args = _ORIGINAL_ADD_MOE_ARGS


def _add_moe_args(parser: ArgumentParser) -> ArgumentParser:
    """Add MUSA-private options after Megatron Core 0.19 options."""

    parser = _ORIGINAL_ADD_MOE_ARGS(parser)

    # Core 0.19 keeps the config field but excludes it from generated CLI options.
    # Its native config builder already copies this field from parsed arguments.
    fp8_group = parser.add_argument_group(title="musa fp8 extensions")
    _add_argument_if_missing(
        parser,
        fp8_group,
        "--tp-only-amax-red",
        action="store_true",
        help="Reduce the FP8 AMAX only in the TP or TP-CP domain.",
    )

    moe_group = parser.add_argument_group(title="musa moe extensions")
    _add_argument_if_missing(
        parser, moe_group, '--use-span-based-attn', action='store_true',
        help='Preserve baseline single-span THD for the local indexed dataset.')
    _add_argument_if_missing(
        parser,
        moe_group,
        "--moe-extended-tp",
        action="store_true",
        help="Deprecated compatibility flag for legacy MUSA launch scripts.",
    )
    _add_argument_if_missing(
        parser,
        moe_group,
        "--moe-use-legacy-grouped-gemm",
        action="store_true",
        help="Use the legacy MUSA grouped-GEMM implementation.",
    )
    _add_argument_if_missing(
        parser,
        moe_group,
        "--moe-device-level-aux-loss-coeff",
        type=float,
        default=None,
        help="Scaling coefficient for the MUSA device-level auxiliary loss.",
    )
    _add_argument_if_missing(
        parser,
        moe_group,
        "--moe-comm-aux-loss-coeff",
        type=float,
        default=None,
        help="Scaling coefficient for the MUSA communication auxiliary loss.",
    )
    _add_argument_if_missing(
        parser,
        moe_group,
        "--moe-device-level-capacity",
        action="store_true",
        help="Apply capacity to an expert device group instead of each expert.",
    )
    _add_argument_if_missing(
        parser,
        moe_group,
        "--moe-complementary-seq-aux-loss",
        action="store_true",
        help="Use the complementary sequence auxiliary loss for DeepSeek-V3.",
    )
    _add_argument_if_missing(
        parser,
        moe_group,
        "--moe-router-norm-topk-prob",
        action="store_true",
        help="Normalize sigmoid top-k routing probabilities.",
    )

    router_group = parser.add_argument_group(title="musa router extensions")
    _add_argument_if_missing(
        parser,
        router_group,
        "--norm-before-router-softmax",
        action="store_true",
        help="Normalize router logits before the score function.",
    )
    _add_argument_if_missing(
        parser,
        router_group,
        "--use-unbiased-norm",
        action="store_true",
        help="Use sample-standard-deviation normalization for router logits.",
    )
    _add_argument_if_missing(
        parser,
        router_group,
        "--moe-router-norm-scale",
        type=float,
        default=1.0,
        help="Scale applied after MUSA router-logit normalization.",
    )

    recompute_group = parser.add_argument_group(title="musa recompute extensions")
    for option, help_text in (
        ("--q-rms-recompute", "Recompute the MLA query up-projection RMSNorm."),
        ("--attn-recompute", "Enable the MUSA attention recompute path."),
        ("--mla-rms-recompute", "Recompute MLA input RMSNorm."),
        ("--mlp-rms-recompute", "Recompute MLP input RMSNorm."),
        (
            "--recompute-variance",
            "Deprecated compatibility option; Core 0.19 rejects it because no "
            "behaviorally equivalent native recompute mode exists.",
        ),
        ("--mlp-recompute", "Recompute grouped-GEMM and dense MLP activations."),
    ):
        _add_argument_if_missing(
            parser,
            recompute_group,
            option,
            action="store_true",
            help=help_text,
        )

    monitor_group = parser.add_argument_group(title="musa moe monitoring")
    for option, help_text in (
        ("--router-prob-var-mointor-freq", "router probability variance"),
        ("--router-logit-var-mointor-freq", "router logit variance"),
        ("--router-maxvio-mointor-freq", "router load-balance violation"),
    ):
        _add_argument_if_missing(
            parser,
            monitor_group,
            option,
            type=int,
            default=0,
            help=f"Logging frequency for {help_text}; zero disables it.",
        )

    offload_group = parser.add_argument_group(title="musa activation offload")
    _add_argument_if_missing(
        parser,
        offload_group,
        "--offload-moe-fc1-input",
        action="store_true",
        help="Offload routed-expert FC1 inputs to CPU memory.",
    )
    _add_argument_if_missing(
        parser,
        offload_group,
        "--offload-moe-fused-swiglu-input",
        action="store_true",
        help="Offload routed-expert fused-SwiGLU inputs to CPU memory.",
    )
    _add_argument_if_missing(
        parser,
        offload_group,
        "--disable-pre-offload-moe-fused-swiglu-input",
        action="store_true",
        help="Disable pre-offload for routed-expert fused-SwiGLU inputs.",
    )

    return parser


_MUSA_CONFIG_DEFAULTS: dict[str, Any] = {
    "moe_extended_tp": False,
    "moe_use_legacy_grouped_gemm": False,
    "moe_device_level_aux_loss_coeff": None,
    "moe_comm_aux_loss_coeff": None,
    "moe_device_level_capacity": False,
    "moe_complementary_seq_aux_loss": False,
    "moe_router_norm_topk_prob": False,
    "q_rms_recompute": False,
    "attn_recompute": False,
    "mla_rms_recompute": False,
    "mlp_rms_recompute": False,
    "recompute_variance": False,
    "mlp_recompute": False,
    "norm_before_router_softmax": False,
    "use_unbiased_norm": False,
    "moe_router_norm_scale": 1.0,
    "router_prob_var_mointor_freq": 0,
    "router_logit_var_mointor_freq": 0,
    "router_maxvio_mointor_freq": 0,
    "offload_moe_fc1_input": False,
    "offload_moe_fused_swiglu_input": False,
}

_ORIGINAL_CORE_CONFIG_FROM_ARGS = getattr(
    argument_utils,
    "_musa_original_core_transformer_config_from_args",
    argument_utils.core_transformer_config_from_args,
)
argument_utils._musa_original_core_transformer_config_from_args = (
    _ORIGINAL_CORE_CONFIG_FROM_ARGS
)


def _translate_legacy_musa_options(args: Any) -> None:
    """Map old MUSA switches to their native Megatron Core 0.19 equivalents."""

    if getattr(args, "recompute_variance", False):
        raise ValueError(
            "--recompute-variance is not supported on Megatron Core 0.19: the "
            "legacy variance-aware whole-forward patch has no behaviorally "
            "equivalent native implementation. Remove the flag and select the "
            "required native modules with --recompute-granularity selective "
            "and --recompute-modules."
        )

    recompute_modules: list[str] = []
    if getattr(args, "recompute_modules", None) is not None:
        recompute_modules.extend(args.recompute_modules)

    requested_recompute_modules = []
    if getattr(args, "attn_recompute", False):
        requested_recompute_modules.append("core_attn")
    if getattr(args, "q_rms_recompute", False):
        requested_recompute_modules.append("mla_up_proj")
    if getattr(args, "mla_rms_recompute", False) or getattr(
        args, "mlp_rms_recompute", False
    ):
        requested_recompute_modules.append("layernorm")
    if getattr(args, "mlp_recompute", False):
        requested_recompute_modules.append("mlp")
        if getattr(args, "num_experts", None) is not None:
            requested_recompute_modules.append("moe")

    if requested_recompute_modules:
        if getattr(args, "recompute_granularity", None) not in (None, "selective"):
            raise ValueError(
                "Legacy MUSA selective-recompute flags cannot be combined with "
                f"recompute_granularity={args.recompute_granularity!r}."
            )
        args.recompute_granularity = "selective"
        for module_name in requested_recompute_modules:
            if module_name not in recompute_modules:
                recompute_modules.append(module_name)
        args.recompute_modules = recompute_modules

    offload_modules: list[str] = []
    if getattr(args, "offload_modules", None) is not None:
        offload_modules.extend(args.offload_modules)
    if getattr(args, "offload_moe_fc1_input", False):
        offload_modules.append("expert_fc1")
    if getattr(args, "offload_moe_fused_swiglu_input", False):
        offload_modules.append("moe_act")
    if offload_modules:
        args.fine_grained_activation_offloading = True
        args.offload_modules = list(dict.fromkeys(offload_modules))


def core_transformer_config_from_args(
    args: Any, config_class: type | None = None
) -> Any:
    """Build the 0.19 config, then attach fields consumed only by MUSA patches."""

    _translate_legacy_musa_options(args)
    config = _ORIGINAL_CORE_CONFIG_FROM_ARGS(args, config_class)
    for name, default in _MUSA_CONFIG_DEFAULTS.items():
        setattr(config, name, getattr(args, name, default))

    config.pre_offload_moe_fused_swiglu_input = not getattr(
        args, "disable_pre_offload_moe_fused_swiglu_input", False
    )
    config.seq_length = getattr(args, "seq_length", getattr(config, "seq_length", None))

    if os.environ.get("ENABLE_HOOK", "0") == "1":
        config.deallocate_pipeline_outputs = False

    if (
        config.norm_before_router_softmax
        and os.environ.get("USE_MUSA_ROUTER", "0") != "1"
    ):
        raise ValueError(
            "--norm-before-router-softmax requires USE_MUSA_ROUTER=1 so the router "
            "extension is actually installed."
        )

    if (
        config.offload_moe_fc1_input or config.offload_moe_fused_swiglu_input
    ) and config.pipeline_model_parallel_size < 3:
        raise ValueError(
            "MUSA MoE activation offload requires pipeline-model-parallel-size >= 3."
        )

    return config


# Patch both locations: arguments.py re-exports the function, while helpers in
# argument_utils.py resolve their module global at call time.
training_arguments._add_moe_args = _add_moe_args
training_arguments.core_transformer_config_from_args = core_transformer_config_from_args
argument_utils.core_transformer_config_from_args = core_transformer_config_from_args

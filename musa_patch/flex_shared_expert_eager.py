"""MUSA-safe Flex/DeepEP shared-expert scheduling.

Megatron's shared-expert overlap mode deliberately disables the TP/SP
collectives embedded in TE Linear. The collectives are not optional: the
caller must run SharedExpertMLP's explicit state machine instead. This patch
does that eagerly after asynchronous Flex dispatch submission and keeps the
shared-expert GEMMs on the current compute stream. It therefore avoids both an
incorrect "call forward after disabling TP/SP" shortcut and a second concurrent
TE GEMM stream, which has proved unsafe on the target MUSA stack.
"""

from functools import wraps

import torch


_INSTALLED = False


def _is_flex_shared_overlap(config):
    """Return whether this patch owns the configuration."""
    return bool(
        getattr(config, "moe_shared_expert_overlap", False)
        and getattr(config, "moe_token_dispatcher_type", None) == "flex"
    )


def _run_explicit_shared_forward(shared_experts, hidden_states):
    """Run every required state in order and return the shared-expert output.

    With the official overlap flag enabled, SharedExpertMLP changes TE Linear's
    ``parallel_mode`` to ``None``. Consequently ``shared_experts(input)`` is no
    longer an equivalent forward: it skips the external sequence-parallel
    AllGather and ReduceScatter. Keep this five-call sequence together so a
    future edit cannot accidentally omit one of those collectives.
    """
    shared_experts.pre_forward_comm(hidden_states)
    shared_experts.linear_fc1_forward_and_act()
    shared_experts.linear_fc2_forward()
    shared_experts.post_forward_comm()
    return shared_experts.get_output()


def install():
    """Install the Flex shared-expert patch once."""
    global _INSTALLED
    if _INSTALLED:
        return

    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.extensions.transformer_engine import te_checkpoint
    from megatron.core.tensor_parallel.mappings import (
        copy_to_tensor_model_parallel_region,
        gather_from_sequence_parallel_region,
    )
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.moe.shared_experts import (
        SharedExpertMLP,
        set_tensor_grad_fn_sequence_sr,
    )
    from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher
    from megatron.core.transformer.transformer_config import TransformerConfig

    original_config_post_init = TransformerConfig.__post_init__
    original_shared_init = SharedExpertMLP.__init__
    original_pre_forward_comm = SharedExpertMLP.pre_forward_comm
    original_moe_forward = MoELayer.forward
    original_moe_backward_dw = MoELayer.backward_dw

    @wraps(original_config_post_init)
    def config_post_init(self):
        # This Megatron revision only rejects the dispatcher pairing. Hide the
        # flag while running the original validator, then restore it and repeat
        # the two other overlap-dependent checks so unsupported combinations
        # are not accidentally admitted.
        target = _is_flex_shared_overlap(self)
        if not target:
            return original_config_post_init(self)

        self.moe_shared_expert_overlap = False
        try:
            result = original_config_post_init(self)
        finally:
            self.moe_shared_expert_overlap = True

        recompute_modules = getattr(self, "recompute_modules", None) or []
        if (
            getattr(self, "recompute_granularity", None) == "selective"
            and "shared_experts" in recompute_modules
        ):
            raise ValueError(
                "shared_experts recompute cannot work with "
                "--moe-shared-expert-overlap."
            )
        assert not getattr(self, "overlap_moe_expert_parallel_comm", False), (
            "disable moe_shared_expert_overlap when enabling "
            "overlap_moe_expert_parallel_comm"
        )
        return result

    @wraps(original_shared_init)
    def shared_init(self, config, *args, **kwargs):
        if _is_flex_shared_overlap(config):
            # The official implementation normally allocates a second compute
            # stream. Reuse the current/default stream on MUSA. Flex/DeepEP
            # still submits communication on its own stream, but two TE GEMM
            # streams are never made concurrent by this patch.
            type(self).stream = torch.cuda.current_stream()
        original_shared_init(self, config, *args, **kwargs)
        if _is_flex_shared_overlap(self.config):
            self.stream = torch.cuda.current_stream()
            type(self).stream = self.stream

    def flex_set_shared_experts(self, shared_experts):
        """Register shared experts; execution is owned by MoELayer below."""
        assert self.config.moe_shared_expert_overlap
        self.shared_experts = shared_experts

    def pre_forward_comm(self, inputs):
        """Start the explicit TP/SP input mapping without a redundant self-wait."""
        if not _is_flex_shared_overlap(self.config):
            return original_pre_forward_comm(self, inputs)
        assert self.cached_output is None
        # shared_init binds self.stream to torch.cuda.current_stream(). The
        # pinned Megatron method first calls self.stream.wait_stream(current),
        # which is unnecessary when both objects are the same stream. Keep the
        # remaining official AllGather/copy behavior unchanged.
        with torch.cuda.stream(self.stream):
            if self.use_shared_expert_gate:
                logits = torch.nn.functional.linear(inputs, self.gate_weight)
                self.gate_score = torch.nn.functional.sigmoid(logits)
            if self.config.sequence_parallel:
                self.cached_fc1_input = gather_from_sequence_parallel_region(
                    inputs, tensor_parallel_output_grad=True
                )
            else:
                self.cached_fc1_input = copy_to_tensor_model_parallel_region(inputs)
            set_tensor_grad_fn_sequence_sr(self.cached_fc1_input, torch.iinfo(torch.int).max)

    @wraps(original_moe_forward)
    def moe_forward(self, hidden_states):
        if not _is_flex_shared_overlap(self.config):
            return original_moe_forward(self, hidden_states)

        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        def custom_forward(inputs):
            # token_dispatch is asynchronous for the tested Flex/DeepEP backend.
            # Submit it first, then run the complete shared-expert TP/SP state
            # machine before routed-expert computation consumes dispatch output.
            shared_input = inputs
            routed_input, probs, residual, routing_info = self.router_and_preprocess(inputs)
            dispatched_input, probs = self.dispatch(routed_input, probs)
            shared_output = _run_explicit_shared_forward(self.shared_experts, shared_input)
            output, mlp_bias = self.routed_experts_compute(
                dispatched_input, probs, residual, routing_info
            )
            return self.combine(output, shared_output), mlp_bias

        if self.moe_layer_recompute:
            if getattr(self.config, "fp8", None) or getattr(self.config, "fp4", None):
                return te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                )
            return tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        return custom_forward(hidden_states)

    @wraps(original_moe_backward_dw)
    def moe_backward_dw(self):
        if not _is_flex_shared_overlap(self.config):
            return original_moe_backward_dw(self)
        self.experts.backward_dw()
        if self.use_shared_expert:
            self.shared_experts.backward_dw()

    TransformerConfig.__post_init__ = config_post_init
    SharedExpertMLP.__init__ = shared_init
    SharedExpertMLP.pre_forward_comm = pre_forward_comm
    MoEFlexTokenDispatcher.set_shared_experts = flex_set_shared_experts
    MoELayer.forward = moe_forward
    MoELayer.backward_dw = moe_backward_dw
    _INSTALLED = True

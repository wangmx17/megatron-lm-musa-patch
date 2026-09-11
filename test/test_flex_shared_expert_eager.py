"""Unit tests for the Flex shared-expert scheduling contract."""

import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / "musa_patch/flex_shared_expert_eager.py"
spec = importlib.util.spec_from_file_location("flex_shared_expert_eager", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class _Config:
    moe_shared_expert_overlap = True
    moe_token_dispatcher_type = "flex"


class _SharedExperts:
    def __init__(self):
        self.calls = []

    def pre_forward_comm(self, hidden_states):
        self.calls.append(("pre_forward_comm", hidden_states))

    def linear_fc1_forward_and_act(self):
        self.calls.append(("linear_fc1_forward_and_act",))

    def linear_fc2_forward(self):
        self.calls.append(("linear_fc2_forward",))

    def post_forward_comm(self):
        self.calls.append(("post_forward_comm",))

    def get_output(self):
        self.calls.append(("get_output",))
        return "shared-output"


class FlexSharedExpertEagerTests(unittest.TestCase):
    def test_target_predicate_requires_both_official_options(self):
        config = _Config()
        self.assertTrue(module._is_flex_shared_overlap(config))
        config.moe_shared_expert_overlap = False
        self.assertFalse(module._is_flex_shared_overlap(config))
        config.moe_shared_expert_overlap = True
        config.moe_token_dispatcher_type = "alltoall"
        self.assertFalse(module._is_flex_shared_overlap(config))

    def test_explicit_tp_sp_state_machine_is_complete_and_ordered(self):
        shared = _SharedExperts()
        output = module._run_explicit_shared_forward(shared, "hidden")
        self.assertEqual(output, "shared-output")
        self.assertEqual(
            shared.calls,
            [
                ("pre_forward_comm", "hidden"),
                ("linear_fc1_forward_and_act",),
                ("linear_fc2_forward",),
                ("post_forward_comm",),
                ("get_output",),
            ],
        )


if __name__ == "__main__":
    unittest.main()

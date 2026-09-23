import importlib.util
from pathlib import Path
import queue
import sys
import types
import unittest
from unittest import mock

import torch


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "musa_patch"
    / "deepep_ace"
    / "ace_wgrad.py"
)


def _load_module():
    class FusedDispatch:
        @staticmethod
        def forward(*args, **kwargs):
            raise NotImplementedError

        @staticmethod
        def backward(*args, **kwargs):
            raise NotImplementedError

        @staticmethod
        def apply(*args, **kwargs):
            raise NotImplementedError

    class MoELayer:
        def __init__(self, *args, **kwargs):
            pass

    class MoEFlexTokenDispatcher:
        pass

    class DeepepManager:
        def dispatch(self, *args, **kwargs):
            raise NotImplementedError

    class EventHandle:
        pass

    class EventOverlap:
        def __init__(self, handle):
            self.handle = handle

    fused_a2a = types.ModuleType("megatron.core.transformer.moe.fused_a2a")
    fused_a2a.EventHandle = EventHandle
    fused_a2a.EventOverlap = EventOverlap
    fused_a2a.FusedDispatch = FusedDispatch
    fused_a2a.get_buffer = lambda *args: None
    fused_a2a.get_hidden_bytes = lambda tensor: tensor.element_size() * tensor.shape[-1]
    fused_a2a.fused_dispatch = lambda *args, **kwargs: None

    token_dispatcher = types.ModuleType(
        "megatron.core.transformer.moe.token_dispatcher"
    )
    token_dispatcher.MoEFlexTokenDispatcher = MoEFlexTokenDispatcher
    token_dispatcher._DeepepManager = DeepepManager
    token_dispatcher.fused_dispatch = lambda *args, **kwargs: None

    moe_layer = types.ModuleType("megatron.core.transformer.moe.moe_layer")
    moe_layer.MoELayer = MoELayer
    moe_layer.get_default_pg_collection = lambda: None

    module_names = (
        "megatron",
        "megatron.core",
        "megatron.core.transformer",
        "megatron.core.transformer.moe",
    )
    modules = {name: types.ModuleType(name) for name in module_names}
    modules.update(
        {
            fused_a2a.__name__: fused_a2a,
            token_dispatcher.__name__: token_dispatcher,
            moe_layer.__name__: moe_layer,
        }
    )
    spec = importlib.util.spec_from_file_location("ace_wgrad_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


_ACE_WGRAD = _load_module()


class _Store:
    def __init__(self):
        self.context = queue.Queue()

    def assert_empty(self):
        if not self.context.empty():
            raise AssertionError("store is not empty")


class _Layer:
    def __init__(self, name):
        self.name = name
        self.wgrad_store = _Store()


def _make_state(events):
    state = _ACE_WGRAD.RoutedFc1Wgrad.__new__(_ACE_WGRAD.RoutedFc1Wgrad)
    state.layers = (_Layer("fc1"), _Layer("fc2"))
    state.layer_names = ("fc1", "fc2")

    def backward_dw(layer):
        layer.wgrad_store.context.get_nowait()
        events.append(layer.name)

    state.native_backward_dw = backward_dw
    state.active = False
    state.next_layer = 0
    return state


class TestAceWgradSchedule(unittest.TestCase):
    def test_fc1_drains_before_wait_and_fc2_after_wait(self):
        events = []
        state = _make_state(events)
        state.begin()
        for layer in state.layers:
            layer.wgrad_store.context.put(object())

        state.drain_before_wait()
        events.append("wait")
        state.drain_after_wait()

        self.assertEqual(events, ["fc1", "wait", "fc2"])
        self.assertFalse(state.active)
        self.assertEqual(state.next_layer, 0)

    def test_queue_shape_is_checked_before_draining(self):
        state = _make_state([])
        state.begin()
        state.layers[0].wgrad_store.context.put(object())

        with self.assertRaisesRegex(RuntimeError, "exactly one routed FC1 and FC2"):
            state.drain_before_wait()

    def test_in_flight_state_cannot_be_reused(self):
        state = _make_state([])
        state.begin()
        with self.assertRaisesRegex(RuntimeError, "in-flight reuse"):
            state.begin()

    def test_dispatch_backward_orders_combine_drain_wait(self):
        events = []

        class AfterEvent:
            def current_stream_wait(self):
                events.append("wait")

        class Buffer:
            def combine(self, *args, **kwargs):
                events.append("combine")
                return torch.zeros(1), torch.zeros(1), AfterEvent()

        class WgradState:
            def drain_before_wait(self):
                events.append("fc1")

            def drain_after_wait(self):
                events.append("fc2")

        ctx = types.SimpleNamespace(
            group=object(),
            handle=object(),
            async_finish=True,
            allocate_on_comm_stream=False,
            wgrad_overlap=WgradState(),
            needs_input_grad=(True,) * 8,
        )
        with mock.patch.object(_ACE_WGRAD, "get_buffer", return_value=Buffer()):
            result = _ACE_WGRAD._fused_dispatch_backward(
                ctx,
                torch.zeros(1, 1),
                None,
                torch.ones(1),
                None,
                None,
            )

        self.assertEqual(events, ["combine", "fc1", "wait", "fc2"])
        self.assertEqual(len(result), 8)


if __name__ == "__main__":
    unittest.main()

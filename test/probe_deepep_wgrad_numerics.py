"""Native MUSA/MATE eager-versus-deferred expert gradient check.

Run on an idle GPU in the training environment with its MATE/Megatron
PYTHONPATH. This checks local gradient math, not distributed communication.
"""
import json
import os

os.environ["USE_DEEPEP_ACE"] = "0"
os.environ["ENABLE_DEEPEP_FC1_WGRAD_OVERLAP"] = "0"

import torch
import musa_patch  # Install the same MATE/TE path as training.
from transformer_engine.pytorch.module.grouped_linear import GroupedLinear
from musa_patch.deepep_wgrad import RoutedFc1Wgrad


def make_pair(delayed):
    layers = [
        GroupedLinear(20, inp, out, bias=False, params_dtype=torch.bfloat16,
                      device="musa", fuse_wgrad_accumulation=True,
                      delay_wgrad_compute=delayed)
        for inp, out in ((2048, 1024), (512, 2048))
    ]
    for layer in layers:
        for weight in layer.parameters():
            weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
            weight.grad_added_to_main_grad = False
    return layers


def forward(layers, x, splits):
    y = layers[0](x, splits)
    if isinstance(y, tuple):
        y = y[0]
    a, b = y.chunk(2, dim=-1)
    y = layers[1](torch.nn.functional.silu(a) * b, splits)
    return y[0] if isinstance(y, tuple) else y


def main():
    torch.musa.set_device(0)
    torch.distributed.init_process_group(
        "gloo", init_method="tcp://127.0.0.1:29959", rank=0, world_size=1
    )
    torch.manual_seed(42)
    torch.musa.manual_seed_all(42)
    eager, delayed = make_pair(False), make_pair(True)
    for src, dst in zip(eager, delayed):
        dst.load_state_dict(src.state_dict())
    state = RoutedFc1Wgrad.__new__(RoutedFc1Wgrad)
    state.layers = tuple(delayed)
    state.layer_names = ("fc1", "fc2")
    state.native_backward_dw = GroupedLinear.backward_dw
    state.active, state.next_layer = False, 0
    results = []

    def compare(name, a, b):
        assert torch.isfinite(a).all().item(), f"non-finite reference: {name}"
        assert torch.isfinite(b).all().item(), f"non-finite candidate: {name}"
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        results.append({"name": name, "max_abs": (a.float() - b.float()).abs().max().item()})

    # Deliberately unequal token counts, including empty experts. The second
    # microbatch must accumulate into, rather than replace, existing main_grad.
    # The third simulates zero_grad at the next optimizer-step boundary.
    for microbatch in range(3):
        if microbatch == 2:
            for layer in eager + delayed:
                for weight in layer.parameters():
                    weight.main_grad.zero_()
                    weight.grad_added_to_main_grad = False
        splits = [0 if i % 7 == 0 else 16 + (i * 13 + microbatch * 7) % 65
                  for i in range(20)]
        x = torch.randn(sum(splits), 2048, device="musa", dtype=torch.bfloat16)
        grad = torch.randn_like(x)
        x0, x1 = x.detach().clone().requires_grad_(), x.detach().clone().requires_grad_()
        state.begin()
        y0, y1 = forward(eager, x0, splits), forward(delayed, x1, splits)
        # Keep this standalone probe on the initialized MUSA host thread.
        # Distributed execution/lifetimes are covered by the real training run.
        with torch.autograd.set_multithreading_enabled(False):
            y0.backward(grad)
            y1.backward(grad)
        state.drain_before_wait()
        state.drain_after_wait()
        torch.musa.synchronize()
        compare(f"mb{microbatch}.output", y0, y1)
        compare(f"mb{microbatch}.dinput", x0.grad, x1.grad)
        for li, (src, dst) in enumerate(zip(eager, delayed)):
            for wi, (w0, w1) in enumerate(zip(src.parameters(), dst.parameters())):
                if splits[wi]:
                    assert w0.main_grad.abs().max().item() > 0, "vacuous zero-gradient test"
                compare(f"mb{microbatch}.fc{li + 1}.expert{wi}.dw", w0.main_grad, w1.main_grad)
    print("NATIVE_WGRAD_NUMERICS=" + json.dumps({"passed": True, "comparisons": results}), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

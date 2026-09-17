"""Regression: monkey-patching must preserve Megatron's config introspection."""
# isort: skip_file
import inspect
import os

os.environ["MUON_TE_EXPERT_BATCH_NS"] = "0"
import musa_patch  # bootstrap MUSA compatibility before importing EO
from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon, _kwargs_from_config
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from musa_patch.muon_expert_batch_v019 import install


def test_factory_kwargs_preserved():
    original_signature = inspect.signature(TensorParallelMuon.__init__)
    config = OptimizerConfig(
        optimizer="muon", lr=2e-5, muon_momentum=0.9,
        muon_nesterov=False, muon_split_qkv=True,
        muon_tp_mode="blockwise", muon_num_ns_steps=5,
        muon_scale_mode="spectral", muon_extra_scale_factor=1.0,
    )
    before = _kwargs_from_config(TensorParallelMuon, "muon", config)
    install()
    after = _kwargs_from_config(TensorParallelMuon, "muon", config)
    assert inspect.signature(TensorParallelMuon.__init__) == original_signature
    assert after == before
    assert after["tp_mode"] == "blockwise"
    assert after["momentum"] == 0.9
    assert after["nesterov"] is False
    assert after["split_qkv"] is True
    assert after["num_ns_steps"] == 5
    assert after["coefficient_type"] == "quintic"
    assert after["scale_mode"] == "spectral"
    assert after["extra_scale_factor"] == 1.0
    install()
    assert _kwargs_from_config(TensorParallelMuon, "muon", config) == before
    print("FACTORY_KWARGS_PRESERVED", after)


if __name__ == "__main__":
    test_factory_kwargs_preserved()

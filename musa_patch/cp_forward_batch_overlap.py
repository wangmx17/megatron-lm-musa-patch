"""Opt-in CP forward batched P2P with device-side consumer waits.

Limited to the validated MUSA TE source and BF16 THD CP4/CP8 configuration.
Installation changes only this process, never the installed TE source files.
"""
import ast
import functools
import hashlib
import inspect
import linecache
import os
from pathlib import Path
import textwrap


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise RuntimeError('TE source guard failed: ' + repr(before[:90]))
    return source.replace(before, after, 1)


def build_source(source, direction):
    source = textwrap.dedent(source)
    source = replace_once(source,
        '                # for req in send_recv_reqs[(i + 1) % 2]:\n                #     req.wait()',
        '                for req in send_recv_reqs[(i + 1) % 2]:\n                    req.wait()')
    source = replace_once(source,
        'send_recv_reqs[i % 2] = flash_attn_p2p_communicate_sync(',
        'send_recv_reqs[i % 2] = flash_attn_p2p_communicate(')
    tree = ast.parse(source)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
    assert all(isinstance(d, ast.Name) and d.id == 'staticmethod' for d in tree.body[0].decorator_list)
    tree.body[0].decorator_list = []
    return ast.unparse(tree) + '\n'


def _install_experiment():
    import transformer_engine.pytorch.attention as attention
    direction = 'forward'
    cls = attention.AttnFuncWithCPAndKVP2P
    original = getattr(cls, direction)
    assert not original.__closure__, 'unexpected closure'
    source = build_source(inspect.getsource(original), direction)
    expected = '2a82ad437680ebed2adfe86cfbb60b7a0a59c83c83d6edbcbb2fa6d0eaa2ead8'
    if hashlib.sha256(source.encode()).hexdigest() != expected:
        raise RuntimeError('Unexpected live TE function; refusing to compose unvalidated patches')
    filename = '<overlap0916_cp_' + direction + '>'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = {}
    exec(compile(source, filename, 'exec'), original.__globals__, namespace)
    setattr(cls, direction, staticmethod(namespace[direction]))
    print('[CP_OVERLAP_INSTALLED]', direction, hashlib.sha256(source.encode()).hexdigest(), flush=True)


TE_SHA256 = '510afa9a3da138697c8c16538efabcb22b8ec5dacaadd6aef5bdd02ff9b510c1'
# CP4/seq65536 and CP8/seq131072 both give 16384 tokens per rank, so the
# validated kernel shapes are identical; only the ring length differs.
VALIDATED_CP_SIZES = (4, 8)


def install():
    """Enable only when MUSA_CP_FORWARD_BATCH_OVERLAP=1 (default off)."""
    if os.environ.get('MUSA_CP_FORWARD_BATCH_OVERLAP', '0') != '1':
        return
    direction = 'forward'
    import torch
    import transformer_engine.pytorch.attention as te

    if os.environ.get('NVTE_BATCH_MHA_P2P_COMM') != '1':
        raise RuntimeError('CP overlap requires NVTE_BATCH_MHA_P2P_COMM=1; separate P2P is not validated')
    if hashlib.sha256(Path(te.__file__).read_bytes()).hexdigest() != TE_SHA256:
        raise RuntimeError('Unsupported TE attention.py source; refusing CP overlap installation')
    cls = te.AttnFuncWithCPAndKVP2P
    original = getattr(cls, direction)
    if getattr(original, '_musa_cp_overlap', False):
        return
    signature = inspect.signature(original)
    _install_experiment()
    candidate = getattr(cls, direction)

    @functools.wraps(original)
    def guarded(*args, **kwargs):
        bound = signature.bind(*args, **kwargs).arguments
        q, k, v = (bound[n] for n in ('q', 'k', 'v'))
        supported = (q.dtype == k.dtype == v.dtype == torch.bfloat16
            and q.device.type == k.device.type == v.device.type == 'musa'
            and tuple(q.shape) == (16384, 16, 128)
            and tuple(k.shape) == tuple(v.shape) == (16384, 1, 128)
            and bound['qkv_format'] == 'thd'
            and bound['attn_mask_type'] == 'padding_causal'
            and bound['attn_bias_type'] == 'no_bias'
            and bound['dropout_p'] == 0 and not bound['fp8']
            and not bound['use_fused_attention']
            and not isinstance(bound['cp_group'], list)
            and te.get_distributed_world_size(bound['cp_group']) in VALIDATED_CP_SIZES)
        if not supported:
            raise RuntimeError('CP overlap is limited to the validated BF16 THD CP4/CP8 MiniCPM5 shape')
        return candidate(*args, **kwargs)

    guarded._musa_cp_overlap = True
    setattr(cls, direction, staticmethod(guarded))
    print('[CP_OVERLAP_GUARDED]', direction, TE_SHA256, flush=True)

"""Opt-in CP backward batched P2P with device-side consumer waits.

Limited to the validated MUSA TE source and BF16 THD CP4 configuration.
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
    # All previous requests finish at the bottom of the prior ring iteration.
    # Current communication reads the previous dKV and writes the other buffer;
    # current FA reads current KV and writes separate dk_/dv_. Wait before dKV accumulation.
    if source.count('flash_attn_p2p_communicate_sync(') != 2:
        raise RuntimeError('unexpected backward P2P layout')
    source = source.replace('flash_attn_p2p_communicate_sync(', 'flash_attn_p2p_communicate(')
    source = replace_once(source, '# async_op=True,', 'async_op=True,')
    source = replace_once(source,
        '        # wait until dKV is received\n        # for req in send_recv_reqs:\n        #     req.wait()',
        '        # Wait only after current FA and dq work has been submitted.\n        for req in send_recv_reqs:\n            req.wait()')
    tree = ast.parse(source)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
    assert all(isinstance(d, ast.Name) and d.id == 'staticmethod' for d in tree.body[0].decorator_list)
    tree.body[0].decorator_list = []
    return ast.unparse(tree) + '\n'


def _install_experiment():
    import transformer_engine.pytorch.attention as attention
    direction = 'backward'
    cls = attention.AttnFuncWithCPAndKVP2P
    original = getattr(cls, direction)
    assert not original.__closure__, 'unexpected closure'
    source = build_source(inspect.getsource(original), direction)
    expected = 'bda4e1789e35847df45ba08decc3e626dddbe25acdace40949c3b452b6f9beaa'
    if hashlib.sha256(source.encode()).hexdigest() != expected:
        raise RuntimeError('Unexpected live TE function; refusing to compose unvalidated patches')
    filename = '<overlap0916_cp_' + direction + '>'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = {}
    exec(compile(source, filename, 'exec'), original.__globals__, namespace)
    setattr(cls, direction, staticmethod(namespace[direction]))
    print('[CP_OVERLAP_INSTALLED]', direction, hashlib.sha256(source.encode()).hexdigest(), flush=True)


TE_SHA256 = '510afa9a3da138697c8c16538efabcb22b8ec5dacaadd6aef5bdd02ff9b510c1'


def install():
    """Enable only when MUSA_CP_BACKWARD_BATCH_OVERLAP=1 (default off)."""
    if os.environ.get('MUSA_CP_BACKWARD_BATCH_OVERLAP', '0') != '1':
        return
    direction = 'backward'
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
        ctx, dout = bound['ctx'], bound['dout']
        supported = (dout.dtype == torch.bfloat16 and dout.device.type == 'musa'
            and tuple(dout.shape) == (16384, 16, 128)
            and ctx.qkv_format == 'thd' and ctx.attn_mask_type == 'padding_causal'
            and ctx.attn_bias_type == 'no_bias' and ctx.dropout_p == 0
            and not ctx.fp8 and not ctx.use_fused_attention
            and ctx.cp_size_a2a == 1
            and te.get_distributed_world_size(ctx.cp_group) == 4)
        if not supported:
            raise RuntimeError('CP overlap is limited to the validated BF16 THD CP4 MiniCPM5 shape')
        return candidate(*args, **kwargs)

    guarded._musa_cp_overlap = True
    setattr(cls, direction, staticmethod(guarded))
    print('[CP_OVERLAP_GUARDED]', direction, TE_SHA256, flush=True)

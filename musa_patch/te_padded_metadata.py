"""Opt-in, narrowly matched TE padded-metadata specialization for MUSA.

Installed TE files and global torch.equal are never modified. Only the two
named padded-metadata comparisons in each supported entry point are changed.
Unexpected source layouts fail before any runtime entry point is replaced.
"""
import ast
import importlib
import inspect
import textwrap


_EXPRESSIONS = {
    ast.dump(ast.parse(
        f"torch.equal(cu_seqlens_{name}_padded[:-1], cu_seqlens_{name}[:-1])",
        mode="eval").body): f"cu_seqlens_{name}_padded"
    for name in ("q", "kv")
}


class _PaddedComparisons(ast.NodeTransformer):
    def __init__(self):
        self.matches = []

    def visit_Call(self, node):
        name = _EXPRESSIONS.get(ast.dump(node))
        if name is None:
            return self.generic_visit(node)
        self.matches.append(name)
        # The surrounding 'is not None and not ...' remains intact. MUSA
        # conservatively selects TE's existing padded-compatible path.
        condition = ast.parse(f'{name}.device.type == "musa"', mode="eval").body
        return ast.copy_location(ast.IfExp(test=condition, body=ast.Constant(False),
                                          orelse=node), node)


def specialize(function):
    raw = inspect.unwrap(function)
    if raw.__code__.co_freevars:
        raise RuntimeError("Unsupported TE function closure; padded optimization not installed")
    lines, start = inspect.getsourcelines(raw)
    tree = ast.parse(textwrap.dedent("".join(lines)))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise RuntimeError("Unsupported TE entry point source")
    definition = tree.body[0]
    # Staticmethod is restored by the owner descriptor below. Preserve all
    # functional decorators, including TE's no_torch_dynamo wrapper.
    definition.decorator_list = [d for d in definition.decorator_list
                                 if not isinstance(d, ast.Name) or d.id != "staticmethod"]
    transform = _PaddedComparisons()
    tree = transform.visit(tree)
    if sorted(transform.matches) != ["cu_seqlens_kv_padded", "cu_seqlens_q_padded"]:
        raise RuntimeError("Unsupported TE padded metadata comparisons; no patch applied")
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, start - 1)
    local = {}
    # Keep the original module globals: TE feature flags can change after import.
    exec(compile(tree, raw.__code__.co_filename, "exec"), raw.__globals__, local)
    replacement = local[definition.name]
    replacement.__module__ = function.__module__
    replacement.__qualname__ = function.__qualname__
    replacement._musa_padded_metadata_no_sync = True
    return replacement


def install():
    core = importlib.import_module("transformer_engine.pytorch.attention")
    musa = importlib.import_module("transformer_engine.musa.pytorch.attention")
    targets = [
        (core.AttnFuncWithCPAndKVP2P, "forward"),
        (core.DotProductAttention, "forward"),
        (musa, "DotProductAttention_forward_before_fa"),
    ]
    flags = [getattr(getattr(owner, name), "_musa_padded_metadata_no_sync", False)
             for owner, name in targets]
    if all(flags):
        return
    if any(flags):
        raise RuntimeError("Partial TE padded metadata installation detected")
    original_adapter = musa.DotProductAttention_forward_before_fa
    if core.DotProductAttention.forward_before_fa is not original_adapter:
        raise RuntimeError("Unsupported TE MUSA adapter alias")
    pending = []
    for owner, name in targets:
        descriptor = inspect.getattr_static(owner, name)
        replacement = specialize(getattr(owner, name))
        pending.append((owner, name, staticmethod(replacement)
                        if isinstance(descriptor, staticmethod) else replacement))
    for owner, name, replacement in pending:
        setattr(owner, name, replacement)
    core.DotProductAttention.forward_before_fa = musa.DotProductAttention_forward_before_fa

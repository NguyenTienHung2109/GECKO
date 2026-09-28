"""GECKO public exports, loaded on demand to keep subsystem imports independent."""

from importlib import import_module

_EXPORTS = {'bitcoin_structural_degree_q4': ('gecko.data.splits.common', 'bitcoin_structural_degree_q4'), 'canonicalize_logical_edges': ('gecko.data.splits.common', 'canonicalize_logical_edges'), 'ScenarioSpec': ('gecko.data.spec', 'ScenarioSpec'), 'build_synthetic_spec': ('gecko.data.datasets.synthetic', 'build_synthetic_spec')}
__all__ = ['ScenarioSpec', 'bitcoin_structural_degree_q4', 'build_synthetic_spec', 'canonicalize_logical_edges']

def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value

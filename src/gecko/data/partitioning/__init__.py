"""GECKO public exports, loaded on demand to keep subsystem imports independent."""

from importlib import import_module

_EXPORTS = {'PersistentSubgraphPartitioner': ('gecko.data.partitioning.materialize', 'PersistentSubgraphPartitioner')}
__all__ = ['PersistentSubgraphPartitioner']

def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value

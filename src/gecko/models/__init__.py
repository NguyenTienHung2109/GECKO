"""GECKO public exports, loaded on demand to keep subsystem imports independent."""

from importlib import import_module

_EXPORTS = {'LegacyGCNAdapter': ('gecko.models.backbones', 'LegacyGCNAdapter'), 'GECKOGraphModel': ('gecko.models.backbones', 'GECKOGraphModel'), 'ModelRegistry': ('gecko.models.registry', 'ModelRegistry')}
__all__ = ['ModelRegistry', 'LegacyGCNAdapter', 'GECKOGraphModel']

def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value

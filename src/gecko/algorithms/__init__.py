"""GECKO public exports, loaded on demand to keep subsystem imports independent."""

from importlib import import_module

_EXPORTS = {'ClientContinualAlgorithm': ('gecko.algorithms.base', 'ClientContinualAlgorithm'), 'ClientMethodContext': ('gecko.algorithms.context', 'ClientMethodContext'), 'MethodRegistry': ('gecko.algorithms.catalog', 'MethodRegistry'), 'TopologyOverlay': ('gecko.algorithms.topology', 'TopologyOverlay')}
__all__ = ['ClientContinualAlgorithm', 'ClientMethodContext', 'MethodRegistry', 'TopologyOverlay']

def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value

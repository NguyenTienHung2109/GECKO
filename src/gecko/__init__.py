"""GECKO public exports, loaded on demand to keep subsystem imports independent."""

from importlib import import_module

__version__ = "0.4.0"
BENCHMARK_VERSION = "1.0.0"

_EXPORTS = {'GECKOConfig': ('gecko.config', 'GECKOConfig'), 'FederatedClient': ('gecko.engine', 'FederatedClient'), 'FederatedCoordinator': ('gecko.engine', 'FederatedCoordinator'), 'MethodRegistry': ('gecko.algorithms.catalog', 'MethodRegistry'), 'ModelRegistry': ('gecko.models.registry', 'ModelRegistry'), 'SUPPORTED_SCENARIOS': ('gecko.data.datasets.registry', 'SUPPORTED_SCENARIOS'), 'ArtifactStore': ('gecko.data.streams', 'ArtifactStore'), 'StreamBuilder': ('gecko.data.streams', 'StreamBuilder'), 'StreamBundle': ('gecko.data.streams', 'StreamBundle'), 'ScenarioSpec': ('gecko.types', 'ScenarioSpec')}
__all__ = ['ArtifactStore', 'FederatedClient', 'FederatedCoordinator', 'MethodRegistry', 'ModelRegistry', 'ScenarioSpec', 'StreamBuilder', 'StreamBundle', 'SUPPORTED_SCENARIOS', 'GECKOConfig']

def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value

"""GECKO public exports, loaded on demand to keep subsystem imports independent."""

from importlib import import_module

_EXPORTS = {'FederatedEvaluator': ('gecko.evaluation.evaluator', 'FederatedEvaluator'), 'summarize_continual_matrix': ('gecko.evaluation.metrics', 'summarize_continual_matrix'), 'LC_STABILITY_POLICY': ('gecko.evaluation.lc_stability', 'LC_STABILITY_POLICY'), 'analyze_lc_metric_stability': ('gecko.evaluation.lc_stability', 'analyze_lc_metric_stability')}
__all__ = ['FederatedEvaluator', 'LC_STABILITY_POLICY', 'analyze_lc_metric_stability', 'summarize_continual_matrix']

def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value

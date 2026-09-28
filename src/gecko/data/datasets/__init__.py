"""Dataset exports with optional graph libraries loaded only on demand."""

from typing import Any as _Any

_DATASETS = {'DGLGNNBenchmarkDataset': 'graph', 'NYCTaxiDataset': 'graph', 'DglGraphPropPredDatasetWithTaskMask': 'graph', 'WikiCSLinkDataset': 'link', 'BitcoinOTCDataset': 'link', 'AromaticityDataset': 'graph', 'TwitchGamerNodeDataset': 'node', 'OgbgPpaSampledDataset': 'graph', 'AskUbuntuDataset': 'link', 'FacebookLinkDataset': 'link', 'SentimentGraphDataset': 'graph', 'ZINCGraphDataset': 'graph', 'AQSOLGraphDataset': 'graph', 'MovielensDataset': 'link'}
# Legacy loaders use wildcard imports, including these shared dependencies.
_COMMON_EXPORTS = (
    "Callable", "Optional", "datetime", "os", "torch", "np", "nx", "F",
    "tqdm", "pickle", "json", "chain", "dgl", "save_graphs", "load_graphs",
    "save_info", "load_info", "makedirs", "download", "extract_archive",
    "DglGraphPropPredDataset", "PubChemBioAssayAromaticity", "MoleculeCSVDataset",
    "smiles_to_bigraph", "pd",
)
__all__ = [*_COMMON_EXPORTS, *_DATASETS]

def __getattr__(name: str) -> _Any:
    """Resolve real-data exports without affecting synthetic-only imports."""

    from importlib import import_module
    if name in _DATASETS:
        module = _DATASETS[name]
    elif name in _COMMON_EXPORTS:
        module = "common"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"gecko.data.datasets.{module}"), name)
    globals()[name] = value
    return value

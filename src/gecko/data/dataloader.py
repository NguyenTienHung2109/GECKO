from __future__ import annotations

from gecko.data.datasets.registry import DATASET_VERSIONS

from dataclasses import replace
import hashlib
from pathlib import Path
from gecko.config import GECKOConfig
from gecko.types import ScenarioSpec
from gecko.data.datasets.synthetic import build_synthetic_spec

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _checksums(paths: list[Path], root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(set(paths))
        if path.is_file()
    }


def _dataset_provenance(dataset: str, save_path: str) -> dict[str, object]:
    from gecko.data.datasets.registry import DATASET_VERSIONS
    root = Path(save_path)
    raw_files: list[Path]
    processed_files: list[Path]
    if dataset == "bitcoin":
        raw_files = [root / "bitcoinotc.csv.gz", root / "bitcoin_metadata_allIL.pkl"]
        processed_files = list(root.glob("bitcoinotc_*/dgl_graph.bin"))
    elif dataset == "cora":
        raw_files = list(root.glob("cora_v2_*/ind.cora_v2.*"))
        processed_files = list(root.glob("cora_v2_*/cora_v2_dgl_graph.*"))
    elif dataset == "citeseer":
        raw_files = list(root.glob("citeseer_*/ind.citeseer.*"))
        processed_files = list(root.glob("citeseer_*/citeseer_dgl_graph.*"))
    elif dataset == "facebook":
        raw_files = [root / "facebook.tar.gz", root / "facebook_metadata_domainIL.pkl"]
        processed_files = list(root.glob("facebook_*/dgl_graph.bin"))
    elif dataset == "ogbn-arxiv":
        dataset_root = root / "ogbn_arxiv"
        raw_files = (
            list((dataset_root / "raw").glob("*"))
            + list((dataset_root / "split").rglob("*.gz"))
            + [dataset_root / "RELEASE_v1.txt"]
        )
        processed_files = [dataset_root / "processed" / "dgl_data_processed"]
    elif dataset == "ogbn-proteins":
        dataset_root = root / "ogbn_proteins"
        raw_files = (
            list((dataset_root / "raw").glob("*"))
            + list((dataset_root / "split").rglob("*.gz"))
            + [dataset_root / "RELEASE_v1.txt", root / "ogbn-proteins_metadata_domainIL.pkl"]
        )
        processed_files = [dataset_root / "processed" / "dgl_data_processed"]
    else:
        raw_files = []
        processed_files = []
    return {
        "dataset_version": DATASET_VERSIONS.get(dataset, "unknown-static-snapshot"),
        "raw_dataset_checksums": _checksums(raw_files, root),
        "processed_cache_checksums": _checksums(processed_files, root),
    }


def load_scenario_spec(config: GECKOConfig) -> ScenarioSpec:
    """Load data only after UEFA configuration validation has succeeded."""

    config.validate()
    scenario = config.scenario
    if scenario.synthetic or scenario.dataset == "synthetic":
        return build_synthetic_spec(
            scenario.problem,
            scenario.incremental_setting,
            num_tasks=scenario.num_tasks,
            num_clients=config.partition.num_clients,
            seed=config.seed,
        )
    metric = scenario.metrics[0]
    loader_num_tasks = (
        scenario.lp_source_num_domains
        if scenario.problem == "LP"
        and scenario.lp_query_split_protocol == "partition_first_query_split_v1"
        else scenario.num_tasks
    )
    common = {
        "dataset_name": scenario.dataset,
        "num_tasks": loader_num_tasks,
        "metric": metric,
        "save_path": scenario.save_path,
        "incr_type": scenario.incremental_setting,
        "task_shuffle": 0,
        "uefa_seed": config.seed,
        "lp_train_negative_ratio": config.training.lp_train_negative_ratio,
    }
    if scenario.problem == "NC":
        from gecko.data.legacy.nodes import NCScenarioLoader

        loader = NCScenarioLoader(**common)
    elif scenario.problem == "LC":
        from gecko.data.legacy.links import LCScenarioLoader

        loader = LCScenarioLoader(**common)
    elif scenario.problem == "LP":
        from gecko.data.legacy.links import LPScenarioLoader

        loader = LPScenarioLoader(**common)
    else:
        raise ValueError(f"Unsupported UEFA problem: {scenario.problem}")
    exported = loader.export_spec()
    if (
        scenario.problem == "LP"
        and scenario.lp_query_split_protocol == "partition_first_query_split_v1"
    ):
        from gecko.data.splits.lp import prepare_partition_first_lp_pool

        exported = prepare_partition_first_lp_pool(
            exported,
            target_num_tasks=scenario.num_tasks,
            source_num_tasks=int(scenario.lp_source_num_domains),
            base_edge_ratio=scenario.lp_base_edge_ratio,
            seed=config.seed,
            domain_mapping=scenario.lp_domain_mapping,
        )
    provenance = _dataset_provenance(scenario.dataset, scenario.save_path)
    constructor_version = exported.metadata.get("constructor_version")
    if constructor_version is None and scenario.domain_constructor is not None:
        constructor_version = 1
    return replace(
        exported,
        metrics=tuple(scenario.metrics),
        metadata={
            **exported.metadata,
            "split_protocol": scenario.split_protocol,
            "feature_provenance": scenario.feature_provenance,
            "lp_candidate_grouping": scenario.lp_candidate_grouping,
            "hits_tie_policy": scenario.hits_tie_policy,
            "lp_protocol_name": scenario.lp_protocol_name,
            "lp_protocol_version": scenario.lp_protocol_version,
            "lp_training_negative_policy": scenario.lp_training_negative_policy,
            "lp_evaluation_candidate_policy": scenario.lp_evaluation_candidate_policy,
            "lp_legacy_score_comparable": scenario.lp_legacy_score_comparable,
            "lp_query_split_protocol": scenario.lp_query_split_protocol,
            "lp_base_topology_policy": scenario.lp_base_topology_policy,
            "lp_base_edge_ratio": scenario.lp_base_edge_ratio,
            "lp_source_num_domains": scenario.lp_source_num_domains,
            "lp_domain_mapping": scenario.lp_domain_mapping,
            "lp_evaluation_negatives_per_client_task": (
                scenario.lp_evaluation_negatives_per_client_task
            ),
            "domain_constructor": scenario.domain_constructor,
            "constructor_version": constructor_version,
            **provenance,
        },
    )




_RELOCATED_EXPORTS = {'DATASET_VERSIONS': ('gecko.data.datasets.registry', 'DATASET_VERSIONS')}

def __getattr__(name: str):
    from importlib import import_module
    if name not in _RELOCATED_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _RELOCATED_EXPORTS[name]
    return getattr(import_module(module), symbol)

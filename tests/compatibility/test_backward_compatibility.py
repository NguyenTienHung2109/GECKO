from __future__ import annotations

import copy
import importlib
import pickle
import subprocess
import sys

import pytest
import torch


def _legacy_dependencies():
    dgl = pytest.importorskip("dgl", reason="legacy BeGin loaders require DGL")
    pytest.importorskip("ogb", reason="legacy scenario modules import OGB eagerly")
    pytest.importorskip("torch_scatter", reason="legacy node scenarios import torch-scatter eagerly")
    return dgl


def _node_dataset(dgl, incremental: str):
    source = torch.arange(24)
    target = (source + 1).remainder(24)
    graph = dgl.graph((torch.cat([source, target]), torch.cat([target, source])))
    graph.ndata["feat"] = torch.randn(24, 4)
    graph.ndata["label"] = torch.arange(24).remainder(4)
    position = torch.arange(24).remainder(3)
    graph.ndata["train_mask"] = position == 0
    graph.ndata["val_mask"] = position == 1
    graph.ndata["test_mask"] = position == 2
    if incremental == "domain":
        graph.ndata["domain"] = torch.arange(24).remainder(2)
    return {"graph": graph, "num_feats": 4, "num_classes": 4}


def _link_classification_dataset(dgl, incremental: str):
    source = torch.arange(24)
    target = (source + 1).remainder(24)
    graph = dgl.graph((source, target))
    graph.ndata["feat"] = torch.randn(24, 4)
    graph.edata["label"] = torch.arange(24).remainder(4)
    position = torch.arange(24).remainder(3)
    graph.edata["train_mask"] = position == 0
    graph.edata["val_mask"] = position == 1
    graph.edata["test_mask"] = position == 2
    if incremental == "domain":
        graph.edata["domain"] = torch.arange(24).remainder(2)
    if incremental == "time":
        graph.edata["time"] = torch.arange(24).remainder(2)
    return {"graph": graph, "num_feats": 4, "num_classes": 4}


def _link_prediction_dataset(dgl, incremental: str):
    source = torch.arange(24)
    target = (source + 1).remainder(24)
    graph = dgl.graph((torch.cat([source, target]), torch.cat([target, source])))
    graph.ndata["feat"] = torch.randn(24, 4)
    task_ids = torch.arange(graph.num_edges()).remainder(2)
    graph.edata[incremental] = task_ids
    tvt = torch.arange(graph.num_edges()).remainder(3)
    negatives = torch.tensor([[node, (node + 7) % 24] for node in range(12)])
    return {
        "graph": graph,
        "num_feats": 4,
        "tvt_splits": tvt,
        "neg_edges": {"val": negatives[:6], "test": negatives[6:]},
    }


def test_41_representative_begin_imports_remain_available():
    assert importlib.import_module("gecko.engine.legacy") is not None
    assert importlib.import_module("gecko") is not None
    assert importlib.import_module("gecko.data") is not None


@pytest.mark.integration
@pytest.mark.parametrize("incremental", ["task", "class", "domain"])
def test_42_existing_nc_task_class_domain_loaders_work_with_custom_data(tmp_path, incremental):
    dgl = _legacy_dependencies()
    from gecko.data.legacy.nodes import NCScenarioLoader

    loader = NCScenarioLoader(
        dataset_name="custom",
        save_path=str(tmp_path),
        num_tasks=2,
        incr_type=incremental,
        metric="accuracy",
        dataset_load_func=lambda save_path: _node_dataset(dgl, incremental),
    )
    assert loader.get_current_dataset() is not None
    before = loader._curr_task
    spec = loader.export_spec()
    assert loader._curr_task == before
    assert spec.problem_type == "NC"
    assert spec.metadata["source"] == "begin.scenarios.nodes.NCScenarioLoader"


@pytest.mark.integration
@pytest.mark.parametrize("incremental", ["task", "class", "time"])
def test_43_existing_lc_task_class_time_behavior_is_backward_compatible(tmp_path, incremental):
    dgl = _legacy_dependencies()
    from gecko.data.legacy.links import LCScenarioLoader

    loader = LCScenarioLoader(
        dataset_name="custom",
        save_path=str(tmp_path),
        num_tasks=2,
        incr_type=incremental,
        metric="accuracy",
        dataset_load_func=lambda save_path: _link_classification_dataset(dgl, incremental),
    )
    current = loader.get_current_dataset()
    assert current is not None
    predictions = torch.zeros(
        int(current.edata["test_mask"].sum()), dtype=torch.long
    )
    loader.next_task(predictions)
    assert loader._curr_task == 1


@pytest.mark.integration
def test_43b_lc_domain_loader_is_backward_compatible_and_hides_domain_metadata(tmp_path):
    dgl = _legacy_dependencies()
    from gecko.data.legacy.links import LCScenarioLoader

    loader = LCScenarioLoader(
        dataset_name="custom",
        save_path=str(tmp_path),
        num_tasks=2,
        incr_type="domain",
        metric="accuracy",
        dataset_load_func=lambda save_path: _link_classification_dataset(dgl, "domain"),
    )
    current = loader.get_current_dataset()
    assert "domain" not in current.edata
    before = loader._curr_task
    spec = loader.export_spec()
    assert loader._curr_task == before
    assert spec.incremental_type == "domain"
    assert spec.metadata["source"] == "begin.scenarios.links.LCScenarioLoader"
    assert spec.domains is not None
    predictions = torch.zeros(
        int(current.edata["test_mask"].sum()), dtype=torch.long
    )
    loader.next_task(predictions)
    assert loader._curr_task == 1


@pytest.mark.integration
def test_43c_bitcoin_metadata_download_preserves_an_existing_cache(
    monkeypatch, tmp_path
):
    dgl = _legacy_dependencies()
    import gecko.data.datasets.link as links

    graph = dgl.graph((torch.arange(12), torch.arange(1, 13).remainder(12)))
    graph.ndata["feat"] = torch.randn(12, 4)
    graph.edata["label"] = torch.zeros(graph.num_edges(), dtype=torch.long)

    class FakeBitcoinDataset:
        def __init__(self, *args, **kwargs):
            pass

        def __getitem__(self, index):
            assert index == 0
            return graph

    download_calls = []

    def fake_download(url, path, **kwargs):
        download_calls.append((url, path, kwargs))
        with open(path, "wb") as handle:
            pickle.dump(
                {"inner_tvt_split": torch.arange(graph.num_edges()) % 10},
                handle,
            )
        return path

    monkeypatch.setattr(links, "BitcoinOTCDataset", FakeBitcoinDataset)
    monkeypatch.setattr(links, "download", fake_download)

    num_classes, num_feats, loaded = links.load_linkc_dataset(
        "bitcoin", None, "task", str(tmp_path)
    )

    assert (num_classes, num_feats) == (6, 4)
    assert loaded is graph
    assert len(download_calls) == 1
    assert download_calls[0][2] == {"overwrite": False}


@pytest.mark.integration
@pytest.mark.parametrize("incremental", ["domain", "time"])
def test_44_existing_lp_domain_time_behavior_is_backward_compatible(tmp_path, incremental):
    dgl = _legacy_dependencies()
    from gecko.data.legacy.links import LPScenarioLoader

    loader = LPScenarioLoader(
        dataset_name="custom",
        save_path=str(tmp_path),
        num_tasks=2,
        incr_type=incremental,
        metric="hits@5",
        dataset_load_func=lambda save_path: _link_prediction_dataset(dgl, incremental),
    )
    current = loader.get_current_dataset()
    assert current is not None
    predictions = torch.zeros_like(current["test"]["label"])
    loader.next_task(predictions)
    assert loader._curr_task == 1


@pytest.mark.integration
def test_44b_begin_gcn_constructs_and_trains_one_step():
    _legacy_dependencies()
    from gecko.models.registry import ModelRegistry

    model = ModelRegistry().create(
        "begin_gcn",
        input_size=4,
        output_size=4,
        hidden_size=8,
        num_layers=2,
        problem_type="NC",
        incremental_type="class",
    )
    source = torch.arange(12)
    target = (source + 1).remainder(12)
    edge_index = torch.stack(
        [torch.cat([source, target]), torch.cat([target, source])]
    )
    features = torch.randn(12, 4)
    queries = torch.arange(12)
    labels = torch.arange(12).remainder(4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    logits = model.forward_queries(features, edge_index, queries, "NC")
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())
    optimizer.step()


@pytest.mark.integration
@pytest.mark.parametrize("problem", ["LC", "LP"])
def test_44c_begin_gcn_matches_original_trainer_self_loop_topology(problem):
    dgl = _legacy_dependencies()
    from gecko.models.backbones import LegacyGCNAdapter

    torch.manual_seed(0)
    adapter = LegacyGCNAdapter(
        input_size=4,
        output_size=3,
        hidden_size=8,
        num_layers=2,
        problem_type=problem,
        incremental_type="domain",
    )
    direct = copy.deepcopy(adapter.original)
    adapter.eval()
    direct.eval()
    source = torch.arange(12)
    target = (source + 1).remainder(12)
    edge_index = torch.stack(
        [torch.cat([source, target]), torch.cat([target, source])]
    )
    features = torch.randn(12, 4)
    queries = torch.tensor([[0, 1], [2, 5], [7, 9]])
    graph = dgl.add_self_loop(
        dgl.graph((edge_index[0], edge_index[1]), num_nodes=12)
    )
    with torch.no_grad():
        adapter_logits = adapter.forward_queries(
            features, edge_index, queries, problem
        )
        if problem == "LC":
            direct_logits = direct(
                graph, features, queries[:, 0], queries[:, 1]
            )
        else:
            hidden = direct(graph, features)
            direct_logits = (
                hidden[queries[:, 0]] * hidden[queries[:, 1]]
            ).sum(dim=1)
    assert torch.equal(adapter_logits, direct_logits)


def test_45_compileall_begin_succeeds():
    completed = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "begin"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

"""The synthetic reviewer smoke must not require optional graph libraries."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_synthetic_cli_without_optional_graph_libraries(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    code = textwrap.dedent(
        """
        import json
        from pathlib import Path
        import sys

        optional = {
            "dgl", "dgllife", "rdkit", "ogb", "torch_geometric",
            "torch_scatter", "torch_sparse", "pymetis", "sklearn",
            "pandas", "cvxpy", "qpth", "quadprog", "wandb",
        }

        for name in optional:
            sys.modules[name] = None
        from gecko.cli.main import main

        config, output = sys.argv[1:]
        assert main(["construct", "--config", config, "--output-root", output]) == 0
        manifests = list(Path(output).rglob("manifest.json"))
        assert len(manifests) == 1
        stream = manifests[0].parent
        assert main(["audit", "--stream", str(stream)]) == 0
        assert main([
            "run", "--config", config, "--stream", str(stream),
            "--output-root", output, "--model", "uefa_gcn",
            "--device", "cpu", "--wandb-mode", "disabled",
        ]) == 0
        results = list((stream / "results").glob("*.json"))
        assert results
        result = json.loads(results[0].read_text())
        assert result["strategy"] == "fedavg"
        assert all(sys.modules.get(name) is None for name in optional)
        """
    )
    environment = {**os.environ, "PYTHONNOUSERSITE": "1", "WANDB_MODE": "disabled"}
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(root / "configs/gecko_v1/scenarios/synthetic_smoke.yaml"),
            str(tmp_path / "streams"),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

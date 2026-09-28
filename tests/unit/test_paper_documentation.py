"""The concise README contains valid, runnable benchmark examples."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shlex

from gecko.benchmarks import paper


def documentation_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    templates = root / "tools/supplementary_templates"
    return templates if templates.exists() else root


def test_readme_benchmark_examples_parse_and_select_supported_cells(capsys) -> None:
    readme = (documentation_root() / "README.md").read_text(encoding="utf-8")
    commands = re.sub(r"\\\r?\n", " ", readme).splitlines()
    prefix = "python -m gecko.benchmarks.paper "
    stages = set()
    for command in commands:
        if not command.strip().startswith(prefix):
            continue
        args = shlex.split(command.strip()[len(prefix):])
        stage_index = args.index("--stage") + 1
        stages.add(args[stage_index])
        # Validate the real CLI and recipes without constructing or training.
        args[stage_index] = "plan"
        assert paper.main(args) == 0
        plan = json.loads(capsys.readouterr().out)
        assert plan["scenarios"]
        assert plan["total_stream_cells"] > 0
    assert {"construct", "run", "test"}.issubset(stages)


def test_documented_entrypoints_and_alignment_guide_are_included() -> None:
    root = documentation_root()
    for relative in ("examples/benchmark.py", "examples/synthetic.py", "docs/paper_alignment.md"):
        assert (root / relative).is_file()
    assert (Path(__file__).resolve().parents[2] / "src/gecko/benchmarks/paper.py").is_file()

from __future__ import annotations

import json
import os
import subprocess
import sys


def _selector_output(python_hash_seed: str) -> list[bool]:
    code = """
import json
import torch
from gecko.data.splits.lp import stable_base_edge_mask
pairs = torch.tensor([[1, 9], [9, 1], [2, 7], [3, 4]], dtype=torch.long)
print(json.dumps(stable_base_edge_mask(
    pairs, seed=17, ratio=0.5, undirected=True
).tolist()))
"""
    environment = {**os.environ, "PYTHONHASHSEED": python_hash_seed}
    output = subprocess.check_output(
        [sys.executable, "-c", code], env=environment, text=True
    )
    return json.loads(output)


def test_base_selector_is_cross_process_and_pythonhashseed_stable():
    first = _selector_output("0")
    second = _selector_output("123456")
    assert first == second == [True, True, False, False]

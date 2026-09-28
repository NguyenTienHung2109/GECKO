from __future__ import annotations

import pytest

from gecko.engine import FederatedCoordinator

from tests.helpers import CASES
from tests.helpers import make_stream


@pytest.mark.parametrize("problem,incremental,num_tasks", CASES)
def test_cpu_end_to_end_all_seven_scenario_families(problem, incremental, num_tasks):
    stream = make_stream(problem, incremental, num_tasks)
    result = FederatedCoordinator(stream, "fedavg", "Bare").run()
    assert result["benchmark_name"] == "UEFA"
    assert result["strategy"] == "fedavg"
    assert result["runtime_seconds"] >= 0
    assert result["summary"]["final_average_performance"] == result["summary"][
        "final_average_performance"
    ]

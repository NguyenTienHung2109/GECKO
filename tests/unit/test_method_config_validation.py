from __future__ import annotations

import copy

import pytest

from gecko.algorithms import method_config as method_config_module
from gecko.algorithms.method_config import METHOD_CONFIG_SCHEMA
from gecko.algorithms.method_config import METHOD_CONFIG_VERSION
from gecko.algorithms.method_config import MethodConfigValidationError
from gecko.algorithms.method_config import UnsupportedMethodConfigError
from gecko.algorithms.method_config import known_method_config_names
from gecko.algorithms.method_config import resolve_method_config
from gecko.algorithms.method_config import validate_method_config


def _legacy(
    *, strategy: str = "fedavg", continual_method: str = "Bare"
) -> dict[str, object]:
    return {
        "schema": METHOD_CONFIG_SCHEMA,
        "version": METHOD_CONFIG_VERSION,
        "name": "legacy_adapter_v1",
        "strategy": {"name": strategy, "parameters": {}},
        "continual_method": {"name": continual_method, "parameters": {}},
    }


def _scaffold(
    *, correction: bool = True, control_updates: bool = True
) -> dict[str, object]:
    return {
        "schema": METHOD_CONFIG_SCHEMA,
        "version": METHOD_CONFIG_VERSION,
        "name": "scaffold_uefa_adam_v1",
        "strategy": {
            "name": "scaffold",
            "parameters": {
                "correction_enabled": correction,
                "control_updates_enabled": control_updates,
            },
        },
        "continual_method": {"name": "Bare", "parameters": {}},
    }


def _twp(*, strategy: str = "local_only", **overrides: object) -> dict[str, object]:
    parameters = {
        "lambda_l": 10000.0,
        "lambda_t": 10000.0,
        "beta": 0.01,
        "middle_layer_index": None,
        "significant_threshold": 1.0e-12,
    }
    parameters.update(overrides)
    return {
        "schema": METHOD_CONFIG_SCHEMA,
        "version": METHOD_CONFIG_VERSION,
        "name": "twp_uefa_v1",
        "strategy": {"name": strategy, "parameters": {}},
        "continual_method": {"name": "TWP", "parameters": parameters},
    }


def _ssm(*, strategy: str = "local_only", **overrides: object) -> dict[str, object]:
    parameters = {
        "sampler_mode": "degree",
        "hop_budgets": [10, 25],
        "replay_weight": 1.0,
        "replay_ceiling_bytes": 16 * 1024 * 1024,
        "stage_count": 8,
    }
    parameters.update(overrides)
    return {
        "schema": METHOD_CONFIG_SCHEMA,
        "version": METHOD_CONFIG_VERSION,
        "name": "ssm_uefa_v1",
        "strategy": {"name": strategy, "parameters": {}},
        "continual_method": {"name": "SSM", "parameters": parameters},
    }


def _known_unsupported(name: str, strategy: str, continual: str):
    strategy_parameters: dict[str, object] = {}
    if name == "fedgta_uefa_v1":
        strategy_parameters = {
            "propagation_steps": 5,
            "propagation_alpha": 0.5,
            "temperature": 20.0,
            "moment_order": 13,
            "moment_type": "origin",
            "similarity_threshold": 0.65,
        }
    return {
        "schema": METHOD_CONFIG_SCHEMA,
        "version": METHOD_CONFIG_VERSION,
        "name": name,
        "strategy": {"name": strategy, "parameters": strategy_parameters},
        "continual_method": {"name": continual, "parameters": {}},
    }


def test_legacy_adapter_is_explicit_and_resolves_existing_registry_eligibility():
    resolved = validate_method_config(
        _legacy(),
        expected_strategy="fedavg",
        expected_continual_method="Bare",
        problem_type="NC",
        incremental_setting="class",
    )
    assert resolved.runnable
    assert resolved.benchmark_eligible
    assert resolved.support_status == "native"
    assert resolved.strategy_name == "fedavg"
    assert resolved.continual_method_name == "Bare"
    assert resolved.strategy_parameters == ()
    assert resolved.continual_method_parameters == ()
    assert resolved.to_dict()["strategy"] == {
        "name": "fedavg",
        "parameters": {},
    }


def test_legacy_adapter_without_scenario_is_runnable_but_conservatively_ineligible():
    resolved = validate_method_config(_legacy(strategy="local_only"))
    assert resolved.runnable
    assert not resolved.benchmark_eligible
    assert resolved.support_status == "runnable_scenario_unresolved"
    assert resolved.problem_type is None


def test_legacy_adapter_uses_method_specific_not_bare_eligibility():
    nc_task = validate_method_config(
        _legacy(continual_method="EWC"),
        problem_type="NC",
        incremental_setting="task",
    )
    assert nc_task.runnable
    assert not nc_task.benchmark_eligible
    assert nc_task.support_status == "implemented_unverified"

    nc_class = validate_method_config(
        _legacy(continual_method="EWC"),
        problem_type="NC",
        incremental_setting="class",
    )
    assert nc_class.runnable
    assert nc_class.benchmark_eligible
    assert nc_class.support_status == "faithful_supported"


@pytest.mark.parametrize("missing", ["schema", "version", "name"])
def test_required_top_level_identity_fields_fail_closed(missing):
    config = _legacy()
    del config[missing]
    with pytest.raises(MethodConfigValidationError, match="missing"):
        resolve_method_config(config)


def test_unknown_top_level_and_component_fields_are_rejected():
    top = _legacy()
    top["notes"] = "must not affect scientific identity"
    with pytest.raises(MethodConfigValidationError, match="unknown"):
        resolve_method_config(top)

    component = _legacy()
    component["strategy"]["implementation"] = "fallback"
    with pytest.raises(MethodConfigValidationError, match="strategy fields"):
        resolve_method_config(component)

    missing_component_field = _legacy()
    del missing_component_field["continual_method"]["parameters"]
    with pytest.raises(MethodConfigValidationError, match="missing"):
        resolve_method_config(missing_component_field)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "uefa-method-config-v1", "schema"),
        ("version", 1, "version"),
        ("version", True, "integer"),
        ("name", "not_a_method", "Unknown v2 method config name"),
    ],
)
def test_schema_version_and_method_name_are_exact(field, value, message):
    config = _legacy()
    config[field] = value
    with pytest.raises(MethodConfigValidationError, match=message):
        resolve_method_config(config)


def test_unknown_strategy_and_continual_method_names_are_rejected():
    strategy = _legacy(strategy="fedaverage_typo")
    with pytest.raises(MethodConfigValidationError, match="Unknown v2 strategy"):
        resolve_method_config(strategy)

    continual = _legacy(continual_method="BareButDifferent")
    with pytest.raises(
        MethodConfigValidationError, match="Unknown v2 continual-method"
    ):
        resolve_method_config(continual)


def test_family_strategy_and_continual_identities_cannot_be_swapped():
    wrong_strategy = _scaffold()
    wrong_strategy["strategy"]["name"] = "fedavg"
    wrong_strategy["strategy"]["parameters"] = {}
    with pytest.raises(
        MethodConfigValidationError, match="does not implement strategy"
    ):
        resolve_method_config(wrong_strategy)

    wrong_continual = _scaffold()
    wrong_continual["continual_method"]["name"] = "TWP"
    with pytest.raises(
        MethodConfigValidationError, match="does not implement continual method"
    ):
        resolve_method_config(wrong_continual)


def test_runtime_expected_identities_are_checked_independently_of_family():
    with pytest.raises(MethodConfigValidationError, match="coordinator"):
        validate_method_config(
            _legacy(),
            expected_strategy="fedprox",
            expected_continual_method="Bare",
        )
    with pytest.raises(MethodConfigValidationError, match="coordinator"):
        validate_method_config(
            _legacy(),
            expected_strategy="fedavg",
            expected_continual_method="EWC",
        )


def test_scaffold_resolves_as_integrated_unverified_adaptation():
    resolved = resolve_method_config(
        _scaffold(),
        expected_strategy="scaffold",
        expected_continual_method="Bare",
        problem_type="NC",
        incremental_setting="class",
    )
    assert resolved.runnable
    assert not resolved.benchmark_eligible
    assert resolved.support_status == "implemented_unverified"
    assert resolved.scientific_fidelity == "mechanism_adaptation"
    assert dict(resolved.strategy_parameters) == {
        "control_updates_enabled": True,
        "correction_enabled": True,
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda config: config["strategy"]["parameters"].pop("correction_enabled"),
        lambda config: config["strategy"]["parameters"].update(
            {"server_learning_rate": 1.0}
        ),
        lambda config: config["strategy"]["parameters"].update(
            {"correction_enabled": 1}
        ),
        lambda config: config["continual_method"]["parameters"].update(
            {"borrow_bare_defaults": True}
        ),
    ],
)
def test_parameter_fields_and_types_are_exact_per_catalog_row(mutate):
    config = _scaffold()
    mutate(config)
    with pytest.raises(MethodConfigValidationError):
        resolve_method_config(config)

    legacy = _legacy()
    legacy["strategy"]["parameters"]["correction_enabled"] = False
    with pytest.raises(MethodConfigValidationError, match="unknown"):
        resolve_method_config(legacy)


def test_disabled_scaffold_mechanisms_are_explicit_diagnostic_metadata():
    resolved = resolve_method_config(_scaffold(correction=False))
    assert resolved.runnable
    assert not resolved.benchmark_eligible
    assert resolved.support_status == "diagnostic_only_disabled_mechanism"
    assert "diagnostic ablation" in resolved.reason


def test_fedgta_bare_nc_class_is_runnable_but_benchmark_ineligible():
    config = _known_unsupported("fedgta_uefa_v1", "fedgta", "Bare")
    resolved = validate_method_config(
        config, problem_type="NC", incremental_setting="class"
    )
    assert resolved.runnable
    assert not resolved.benchmark_eligible
    assert resolved.support_status == "implemented_unverified"
    assert resolved.scientific_fidelity == "mechanism_adaptation"


def test_fedgta_ssm_nc_class_is_only_an_interaction_pilot():
    config = _known_unsupported("fedgta_uefa_v1", "fedgta", "SSM")
    config["continual_method"]["parameters"] = {
        "sampler_mode": "uniform",
        "hop_budgets": [10, 25],
        "replay_weight": 1.0,
        "replay_ceiling_bytes": 16 * 1024 * 1024,
        "stage_count": 8,
    }
    resolved = validate_method_config(
        config, problem_type="NC", incremental_setting="class"
    )
    assert resolved.runnable
    assert not resolved.benchmark_eligible
    assert resolved.support_status == "implemented_unverified_interaction_pilot"
    assert resolved.scientific_fidelity == "mechanism_adaptation_interaction_pilot"
    assert dict(resolved.continual_method_parameters)["sampler_mode"] == "uniform"


def test_fedgta_twp_nc_class_is_only_an_interaction_pilot():
    config = _known_unsupported("fedgta_uefa_v1", "fedgta", "TWP")
    config["continual_method"]["parameters"] = {
        "lambda_l": 10000.0,
        "lambda_t": 10000.0,
        "beta": 0.01,
        "middle_layer_index": None,
        "significant_threshold": 1.0e-12,
    }
    resolved = validate_method_config(
        config, problem_type="NC", incremental_setting="class"
    )
    assert resolved.runnable
    assert not resolved.benchmark_eligible
    assert resolved.support_status == "implemented_unverified_interaction_pilot"
    assert resolved.scientific_fidelity == "mechanism_adaptation_interaction_pilot"
    assert dict(resolved.continual_method_parameters)["lambda_l"] == 10000.0


def test_fedgta_dslr_nc_class_is_only_an_interaction_pilot():
    config = _known_unsupported("fedgta_uefa_v1", "fedgta", "DSLR")
    config["continual_method"]["parameters"] = {
        "beta": 0.1,
        "radius": 0.2,
        "structure_lambda": 0.5,
        "top_n": 5,
        "candidate_k": 50,
        "tau": 0.8,
        "structure_epochs": 99,
        "structure_learning_rate": 0.01,
        "structure_hidden_dim": 64,
        "structure_heads": 4,
        "replay_fraction": 0.05,
        "replay_ceiling_bytes": 16 * 1024 * 1024,
        "undirected": True,
        "selection_mode": "coverage_diversity",
        "structure_mode": "full",
    }
    resolved = validate_method_config(
        config, problem_type="NC", incremental_setting="class"
    )
    assert resolved.runnable
    assert not resolved.benchmark_eligible
    assert resolved.support_status == "implemented_unverified_interaction_pilot"
    assert resolved.scientific_fidelity == "mechanism_adaptation_interaction_pilot"
    assert dict(resolved.continual_method_parameters)["structure_epochs"] == 99


@pytest.mark.parametrize(
    "incremental,continual",
    [("domain", "Bare"), ("task", "DSLR")],
)
def test_fedgta_unsupported_scenarios_and_compositions_fail_closed(
    incremental: str, continual: str
):
    config = _known_unsupported("fedgta_uefa_v1", "fedgta", continual)
    if continual == "DSLR":
        config["continual_method"]["parameters"] = {
            "beta": 0.1,
            "radius": 0.2,
            "structure_lambda": 0.5,
            "top_n": 5,
            "candidate_k": 50,
            "tau": 0.8,
            "structure_epochs": 99,
            "structure_learning_rate": 0.01,
            "structure_hidden_dim": 64,
            "structure_heads": 4,
            "replay_fraction": 0.05,
            "replay_ceiling_bytes": 16 * 1024 * 1024,
            "undirected": True,
            "selection_mode": "coverage_diversity",
            "structure_mode": "full",
        }
    resolved = resolve_method_config(
        config, problem_type="NC", incremental_setting=incremental
    )
    assert not resolved.runnable
    assert not resolved.benchmark_eligible
    with pytest.raises(UnsupportedMethodConfigError):
        validate_method_config(
            config, problem_type="NC", incremental_setting=incremental
        )



@pytest.mark.parametrize(
    "field,value",
    [
        ("propagation_steps", 4),
        ("propagation_alpha", 0.4),
        ("temperature", 10.0),
        ("moment_order", 12),
        ("moment_type", "unsupported"),
        ("similarity_threshold", 0.7),
    ],
)
def test_fedgta_requires_the_declared_paper_grid(field: str, value: object):
    config = _known_unsupported("fedgta_uefa_v1", "fedgta", "Bare")
    config["strategy"]["parameters"][field] = value
    with pytest.raises(MethodConfigValidationError):
        resolve_method_config(config)


def test_twp_is_explicit_runnable_ineligible_and_cannot_be_relabelled_bare():
    twp = _twp()
    resolved = resolve_method_config(
        twp, problem_type="NC", incremental_setting="class"
    )
    assert resolved.runnable
    assert not resolved.benchmark_eligible
    assert resolved.support_status == "implemented_unverified"
    assert resolved.scientific_fidelity == "mechanism_adaptation"
    assert dict(resolved.continual_method_parameters) == {
        "beta": 0.01,
        "lambda_l": 10000.0,
        "lambda_t": 10000.0,
        "middle_layer_index": None,
        "significant_threshold": 1.0e-12,
    }

    relabelled = copy.deepcopy(twp)
    relabelled["continual_method"]["name"] = "Bare"
    with pytest.raises(
        MethodConfigValidationError, match="does not implement continual method"
    ):
        resolve_method_config(relabelled)


def test_ssm_is_explicit_nc_class_only_ineligible_and_node_only_is_diagnostic():
    assert method_config_module._FAMILIES["ssm_uefa_v1"].continual_parameter_fields == (
        "hop_budgets",
        "replay_ceiling_bytes",
        "replay_weight",
        "sampler_mode",
        "stage_count",
    )
    resolved = resolve_method_config(
        _ssm(), problem_type="NC", incremental_setting="class"
    )
    assert resolved.runnable
    assert not resolved.benchmark_eligible
    assert resolved.support_status == "implemented_unverified"
    assert resolved.scientific_fidelity == "mechanism_adaptation"
    assert dict(resolved.continual_method_parameters) == {
        "hop_budgets": (10, 25),
        "replay_ceiling_bytes": 16 * 1024 * 1024,
        "replay_weight": 1.0,
        "sampler_mode": "degree",
        "stage_count": 8,
    }

    diagnostic = resolve_method_config(
        _ssm(hop_budgets=[0, 0]),
        problem_type="NC",
        incremental_setting="class",
    )
    assert diagnostic.runnable
    assert diagnostic.support_status == "diagnostic_only_node_only_ablation"

    task_adaptation = resolve_method_config(
        _ssm(), problem_type="NC", incremental_setting="task"
    )
    assert task_adaptation.runnable
    assert task_adaptation.support_status == "implemented_unverified_task_adaptation"


def test_fedpub_twp_nc_class_is_only_an_interaction_pilot():
    config = {
        "schema": METHOD_CONFIG_SCHEMA,
        "version": METHOD_CONFIG_VERSION,
        "name": "fed_pub_uefa_v1",
        "strategy": {
            "name": "fed_pub",
            "parameters": {
                "tau": 3.0,
                "lambda1": 0.001,
                "lambda2": 0.001,
                "proxy_seed": 20240718,
            },
        },
        "continual_method": {
            "name": "TWP",
            "parameters": {
                "lambda_l": 10000.0,
                "lambda_t": 10000.0,
                "beta": 0.01,
                "middle_layer_index": None,
                "significant_threshold": 1.0e-12,
            },
        },
    }
    resolved = validate_method_config(
        config, problem_type="NC", incremental_setting="class"
    )
    assert resolved.runnable
    assert not resolved.benchmark_eligible
    assert resolved.support_status == "implemented_unverified_interaction_pilot"
    assert resolved.scientific_fidelity == "mechanism_adaptation_interaction_pilot"
    assert dict(resolved.continual_method_parameters)["lambda_l"] == 10000.0


def test_fedpub_dslr_nc_class_is_only_an_interaction_pilot():
    config = {
        "schema": METHOD_CONFIG_SCHEMA,
        "version": METHOD_CONFIG_VERSION,
        "name": "fed_pub_uefa_v1",
        "strategy": {
            "name": "fed_pub",
            "parameters": {
                "tau": 3.0,
                "lambda1": 0.001,
                "lambda2": 0.001,
                "proxy_seed": 20240718,
            },
        },
        "continual_method": {
            "name": "DSLR",
            "parameters": {
                "beta": 0.1,
                "radius": 0.2,
                "structure_lambda": 0.5,
                "top_n": 5,
                "candidate_k": 50,
                "tau": 0.8,
                "structure_epochs": 99,
                "structure_learning_rate": 0.01,
                "structure_hidden_dim": 64,
                "structure_heads": 4,
                "replay_fraction": 0.05,
                "replay_ceiling_bytes": 16 * 1024 * 1024,
                "undirected": True,
                "selection_mode": "coverage_diversity",
                "structure_mode": "full",
            },
        },
    }
    resolved = validate_method_config(
        config, problem_type="NC", incremental_setting="class"
    )
    assert resolved.runnable
    assert not resolved.benchmark_eligible
    assert resolved.support_status == "implemented_unverified_interaction_pilot"
    assert resolved.scientific_fidelity == "mechanism_adaptation_interaction_pilot"
    assert dict(resolved.continual_method_parameters)["structure_epochs"] == 99


def test_scaffold_non_nc_scenario_fails_closed_with_resolution_metadata():
    resolved = resolve_method_config(
        _scaffold(), problem_type="LC", incremental_setting="class"
    )
    assert not resolved.runnable
    assert resolved.support_status == "unsupported_scientific"
    with pytest.raises(UnsupportedMethodConfigError) as raised:
        validate_method_config(
            _scaffold(), problem_type="LC", incremental_setting="class"
        )
    assert raised.value.resolution.problem_type == "LC"


def test_scenario_identity_must_be_complete_and_known():
    with pytest.raises(MethodConfigValidationError, match="supplied together"):
        resolve_method_config(_legacy(), problem_type="NC")
    with pytest.raises(MethodConfigValidationError, match="Unknown UEFA scenario"):
        resolve_method_config(
            _legacy(), problem_type="NC", incremental_setting="made_up"
        )


def test_catalog_lists_supported_and_fail_closed_future_families():
    names = known_method_config_names()
    assert names == tuple(sorted(names))
    assert "legacy_adapter_v1" in names
    assert "scaffold_uefa_adam_v1" in names
    assert {
        "twp_uefa_v1",
        "ssm_uefa_v1",
        "dslr_uefa_v1",
        "fedgta_uefa_v1",
        "fed_pub_uefa_v1",
        "feddc_uefa_v1",
        "power_uefa_v1",
    }.issubset(names)

"""Registry for UEFA and statically discovered original BeGin graph models."""

from __future__ import annotations

import ast
import inspect
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Tuple

from gecko.models.backbones import LegacyGCNAdapter
from gecko.models.backbones import GECKOGraphModel
from gecko.models.fedfst_gat import FedFSTGAT


_MODEL_DISCOVERY_EXCLUDED_MODULE_PREFIXES = (
    "begin.algorithms.fedfst",
)


@dataclass(frozen=True)
class ModelEntry:
    canonical_name: str
    source_module: str
    supported_problem_types: Tuple[str, ...]
    supported_algorithms: Tuple[str, ...]
    constructor_signature: str
    output_semantics: str
    task_mask_requirements: str
    parameter_sharing_policy: str
    smoke_test_status: str
    component_kind: str
    has_factory: bool
    runnable_by_default: bool
    benchmark_eligible: bool
    benchmark_tier: str
    scientific_fidelity: str
    release_status: str
    state_classification: str
    evidence: Tuple[str, ...]


class ModelRegistry:
    """Expose the UEFA model and every original model class without eager imports."""

    def __init__(self) -> None:
        self._entries: Dict[str, ModelEntry] = {}
        self._factories: Dict[str, Callable[..., Any]] = {}
        self.register(
            ModelEntry(
                canonical_name="fedfst_gat",
                source_module="begin.uefa.models.fedfst_gat",
                supported_problem_types=("NC",),
                supported_algorithms=("Bare",),
                constructor_signature=str(inspect.signature(FedFSTGAT)),
                output_semantics="fixed full-dimensional NC head",
                task_mask_requirements="external immutable global-task masks",
                parameter_sharing_policy="all trainable parameters shared; no buffers",
                smoke_test_status="paper_profile_oracle_and_real_nc_class_task_full",
                component_kind="graph_model_adapter",
                has_factory=True,
                runnable_by_default=True,
                benchmark_eligible=True,
                benchmark_tier="core",
                scientific_fidelity="paper_linked_fedfst_gat_profile",
                release_status="supported",
                state_classification=(
                    "all trainable parameters are shareable; no persistent buffers "
                    "or continual classifier state"
                ),
                evidence=(
                    "bit-exact train/eval oracle against paper-linked PyG GAT",
                    "factory/forward/backward/aggregation/snapshot/CUDA tests",
                    "fixed ogbn-arxiv NC-Class and NC-Task eight-stage CUDA smokes",
                ),
            ),
            FedFSTGAT,
        )
        self.register(
            ModelEntry(
                canonical_name="uefa_gcn",
                source_module="begin.uefa.models.adapters",
                supported_problem_types=("NC", "LC", "LP"),
                supported_algorithms=("Bare",),
                constructor_signature=str(inspect.signature(GECKOGraphModel)),
                output_semantics="fixed full-dimensional NC/LC head; dot-product LP",
                task_mask_requirements="external immutable global-task masks",
                parameter_sharing_policy="encoder and predictor trainable parameters shared",
                smoke_test_status="tested_synthetic",
                component_kind="graph_model",
                has_factory=True,
                runnable_by_default=False,
                benchmark_eligible=False,
                benchmark_tier="extended_reference",
                scientific_fidelity="dependency_light_reference_only",
                release_status="reference_only",
                state_classification=(
                    "trainable parameters are shareable; no persistent buffers or "
                    "continual classifier state"
                ),
                evidence=("synthetic constructor/forward/backward/E2E tests",),
            ),
            GECKOGraphModel,
        )
        self.register(
            ModelEntry(
                canonical_name="begin_gcn",
                source_module="begin.utils.models",
                supported_problem_types=("NC", "LC", "LP"),
                supported_algorithms=("Bare", "LwF", "EWC", "MAS", "ERGNN"),
                constructor_signature=str(inspect.signature(LegacyGCNAdapter)),
                output_semantics="original GCNNode/GCNLink encoder with fixed external masking",
                task_mask_requirements="external immutable global-task masks",
                parameter_sharing_policy=(
                    "trainable floating parameters shared; BatchNorm buffers retained "
                    "per client"
                ),
                smoke_test_status="verified_k1_parity_and_real_k10",
                component_kind="graph_model_adapter",
                has_factory=True,
                runnable_by_default=True,
                benchmark_eligible=True,
                benchmark_tier="core",
                scientific_fidelity="verified_original_begin_engine",
                release_status="supported",
                state_classification=(
                    "shared=trainable floating parameters; local=BatchNorm buffers, "
                    "AdaptiveLinear observed/output masks, and DGL graph cache"
                ),
                evidence=(
                    "K=1 NC/LC/LP engine and trainer parity",
                    "Five-seed real K10 execution across all seven families",
                    "Constructor/forward/backward/masking/state-policy tests",
                ),
            ),
            LegacyGCNAdapter,
        )
        self._discover_original_models()

    def register(self, entry: ModelEntry, factory: Callable[..., Any] | None = None) -> None:
        self._entries[entry.canonical_name] = entry
        if factory is not None:
            self._factories[entry.canonical_name] = factory

    def _discover_original_models(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        models_root = package_root / "models" / "legacy"
        algorithms_root = package_root / "algorithms" / "continual" / "legacy"
        sources = list(sorted(models_root.glob("models*.py")))
        sources.append(models_root / "pretraining.py")
        sources.extend(source for source in sorted(algorithms_root.rglob("*.py")) if source.name != "__init__.py")
        for source in sources:
            if not source.exists():
                continue
            if source.is_relative_to(models_root):
                module = "begin.utils." + ".".join(source.relative_to(models_root).with_suffix("").parts)
            else:
                module = "begin.algorithms." + ".".join(source.relative_to(algorithms_root).with_suffix("").parts)
            if any(
                module == prefix or module.startswith(f"{prefix}.")
                for prefix in _MODEL_DISCOVERY_EXCLUDED_MODULE_PREFIXES
            ):
                continue
            tree = ast.parse(source.read_text(encoding="utf-8"))
            classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
            model_classes = {
                node.name
                for node in classes
                if any("Module" in ast.unparse(base) for base in node.bases)
                or node.name.lower().startswith(("gcn", "fullgcn", "adaptive", "progressive"))
            }
            changed = True
            while changed:
                changed = False
                for node in classes:
                    bases = {ast.unparse(base).rsplit(".", 1)[-1] for base in node.bases}
                    if node.name not in model_classes and bases.intersection(model_classes):
                        model_classes.add(node.name)
                        changed = True
            for node in classes:
                if node.name not in model_classes:
                    continue
                name = f"original:{module}.{node.name}"
                if name in self._entries:
                    continue
                supported_problems = self._infer_problem_types(module, node.name)
                algorithm = self._infer_algorithm(module)
                self.register(
                    ModelEntry(
                        canonical_name=name,
                        source_module=module,
                        supported_problem_types=supported_problems,
                        supported_algorithms=(algorithm,),
                        constructor_signature=self._ast_signature(node),
                        output_semantics="original BeGin implementation",
                        task_mask_requirements="original AdaptiveLinear/task contract",
                        parameter_sharing_policy="requires UEFA adapter review",
                        smoke_test_status="ast_discovered_no_factory",
                        component_kind=self._infer_component_kind(module, node.name),
                        has_factory=False,
                        runnable_by_default=False,
                        benchmark_eligible=False,
                        benchmark_tier="unsupported_discovery",
                        scientific_fidelity="not_verified",
                        release_status="discovered_not_runnable",
                        state_classification="not audited; no UEFA factory",
                        evidence=("AST discovery record only",),
                    )
                )

    @staticmethod
    def _ast_signature(node: ast.ClassDef) -> str:
        initializer = next(
            (
                item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == "__init__"
            ),
            None,
        )
        if initializer is None:
            return "()"
        arguments = initializer.args
        positional = [arg.arg for arg in arguments.posonlyargs + arguments.args]
        if positional and positional[0] == "self":
            positional = positional[1:]
        defaults = [None] * (len(positional) - len(arguments.defaults)) + list(arguments.defaults)
        rendered = [
            name if default is None else f"{name}={ast.unparse(default)}"
            for name, default in zip(positional, defaults)
        ]
        if arguments.vararg is not None:
            rendered.append(f"*{arguments.vararg.arg}")
        rendered.extend(
            f"{arg.arg}={ast.unparse(default)}"
            for arg, default in zip(arguments.kwonlyargs, arguments.kw_defaults)
            if default is not None
        )
        if arguments.kwarg is not None:
            rendered.append(f"**{arguments.kwarg.arg}")
        return f"({', '.join(rendered)})"

    @staticmethod
    def _infer_problem_types(module: str, class_name: str) -> Tuple[str, ...]:
        final_component = module.rsplit(".", 1)[-1]
        if final_component == "nodes" or class_name.endswith("Node"):
            return ("NC",)
        if final_component == "links" or "Link" in class_name or class_name.endswith("Edge"):
            return ("LC", "LP")
        if final_component == "graphs" or class_name.endswith("Graph"):
            return ("GC",)
        return ("NC", "LC", "LP", "GC")

    @staticmethod
    def _infer_algorithm(module: str) -> str:
        parts = module.split(".")
        if "algorithms" not in parts:
            return "*"
        raw = parts[parts.index("algorithms") + 1]
        aliases = {"cat": "CaT", "cgnn": "CGNN", "ergnn": "ERGNN", "pignn": "PIGNN"}
        return aliases.get(raw, raw.capitalize())

    @staticmethod
    def _infer_component_kind(module: str, class_name: str) -> str:
        lowered = class_name.lower()
        if module.endswith("pretraining"):
            return "pretraining_component"
        if "sampler" in lowered:
            return "sampler"
        if "discriminator" in lowered:
            return "algorithm_internal_component"
        if lowered.endswith("conv") or "linear" in lowered:
            return "layer"
        if module.startswith("begin.algorithms"):
            return "algorithm_internal_component"
        return "discovered_graph_component"

    def names(self) -> Tuple[str, ...]:
        """Return only models with verified default UEFA execution."""

        return tuple(
            sorted(
                name
                for name, entry in self._entries.items()
                if entry.runnable_by_default
            )
        )

    def factory_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._factories))

    def discovered_names(self) -> Tuple[str, ...]:
        return tuple(
            sorted(name for name, entry in self._entries.items() if not entry.has_factory)
        )

    def all_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._entries))

    def entries(self) -> Tuple[ModelEntry, ...]:
        return tuple(self._entries[name] for name in self.all_names())

    def entry(self, name: str) -> ModelEntry:
        """Return immutable support metadata for one registered model/component."""

        try:
            return self._entries[name]
        except KeyError as error:
            raise ValueError(f"Unknown model or component {name!r}.") from error

    def create(self, name: str, **kwargs: Any) -> Any:
        if name not in self._factories:
            raise ValueError(f"Model {name!r} has no dependency-safe UEFA factory.")
        return self._factories[name](**kwargs)

    def as_records(self) -> list[Dict[str, Any]]:
        return [asdict(entry) for entry in self.entries()]

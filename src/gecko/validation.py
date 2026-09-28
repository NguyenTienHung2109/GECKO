"""Validation errors and shared validation helpers."""

from __future__ import annotations

from typing import Iterable


class UEFAError(RuntimeError):
    """Base error for UEFA."""


class ConfigurationError(UEFAError, ValueError):
    """Raised before loading data when a configuration is invalid."""


class ScenarioValidationError(UEFAError, ValueError):
    """Raised when a scenario specification violates its schema."""


class PartitionInfeasibleError(UEFAError):
    """Raised when dense client-task support cannot be satisfied."""

    def __init__(self, message: str, failures: Iterable[str] = ()) -> None:
        self.failures = tuple(failures)
        detail = "\n".join(f"- {failure}" for failure in self.failures)
        super().__init__(message if not detail else f"{message}\n{detail}")


class ArtifactIntegrityError(UEFAError):
    """Raised when a stream checksum or manifest is invalid."""


class AggregationError(UEFAError):
    """Raised when client updates are not safely aggregatable."""


class DatasetDesignError(UEFAError, ValueError):
    """Raised when a stream violates a predeclared scientific design rule."""


class UnsupportedCombinationError(ConfigurationError):
    """Raised for a problem/incremental-setting combination outside UEFA v1."""

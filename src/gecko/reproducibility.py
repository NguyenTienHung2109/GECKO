"""Local deterministic random-number utilities."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

import torch


def derive_seed(base_seed: int, *parts: Any) -> int:
    """Derive a stable 63-bit seed without touching global RNG state."""

    payload = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def torch_generator(base_seed: int, *parts: Any) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derive_seed(base_seed, *parts))
    return generator


def python_random(base_seed: int, *parts: Any) -> random.Random:
    return random.Random(derive_seed(base_seed, *parts))


def repository_root() -> Path:
    """Locate the source checkout without depending on module nesting depth."""
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        source = parent / "src" / "gecko"
        if (parent / "pyproject.toml").is_file() and source.is_dir() and module_path.is_relative_to(source):
            return parent
    # A wheel installed into a venv nested inside another checkout must not
    # accidentally resolve configs from that unrelated ancestor checkout.
    for parent in (Path.cwd(), *Path.cwd().parents):
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "gecko").is_dir():
            return parent
    raise ValueError("GECKO recipes require the extracted source directory; run from it or install it with pip install -e .")

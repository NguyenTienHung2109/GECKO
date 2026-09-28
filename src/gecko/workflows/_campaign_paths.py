"""Existing campaign paths and scenario sets."""
from pathlib import Path
from gecko.reproducibility import repository_root

ROOT = repository_root()
DEFAULT_STORE = Path("generated_streams/lpt_dirichlet_temporal_v4")
DEFAULT_OUTPUT = Path("artifacts/lpt_dirichlet_temporal_v4")
NC_ALL = frozenset({("NC", "task"), ("NC", "class"), ("NC", "domain")})
NC_TASK = frozenset({("NC", "task")})
NC_TASK_CLASS = frozenset({("NC", "task"), ("NC", "class")})
NC_CLASS = frozenset({("NC", "class")})
LC_ALL = frozenset({("LC", "task"), ("LC", "class"), ("LC", "domain")})
LP_DOMAIN = frozenset({("LP", "domain")})

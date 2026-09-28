"""Read historical scientific config values with GECKO validation."""
from gecko.config import GECKOConfig
from gecko.compat.paths import resolve_source_path


def load_config(path):
    return GECKOConfig.from_yaml(resolve_source_path(path))

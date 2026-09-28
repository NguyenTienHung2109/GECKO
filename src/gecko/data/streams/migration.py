from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from gecko.data.streams.builder import stream_identity

def migrate_stream_to_v2(
    path: str | Path,
    output_root: str | Path,
    *,
    repository_root: str | Path = ".",
    trusted_legacy: bool = False,
) -> Path:
    """Repackage a verified schema-v1 stream without changing tensor semantics."""
    from gecko.data.streams.store import load_stream
    from gecko.data.streams.store import load_trusted_legacy_stream
    from gecko.data.streams.store import save_stream

    bundle = (
        load_trusted_legacy_stream(path)
        if trusted_legacy
        else load_stream(path)
    )
    config = replace(
        bundle.config,
        benchmark_schema_version=2,
        output_root=str(output_root),
    )
    stream_id, stream_hash = stream_identity(config)
    migrated = replace(
        bundle,
        config=config,
        stream_id=stream_id,
        stream_hash=stream_hash,
    )
    return save_stream(
        migrated,
        output_root,
        repository_root=repository_root,
    )



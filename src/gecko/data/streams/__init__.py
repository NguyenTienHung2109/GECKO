"""GECKO public exports, loaded on demand to keep subsystem imports independent."""

from importlib import import_module

_EXPORTS = {'ArtifactStore': ('gecko.data.streams.store', 'ArtifactStore'), 'audit_stream': ('gecko.data.streams.store', 'audit_stream'), 'derive_order_variant': ('gecko.data.streams.store', 'derive_order_variant'), 'derive_order_variants': ('gecko.data.streams.store', 'derive_order_variants'), 'derive_participation_variant': ('gecko.data.streams.store', 'derive_participation_variant'), 'gc_objects': ('gecko.data.streams.store', 'gc_objects'), 'load_stream': ('gecko.data.streams.store', 'load_stream'), 'load_trusted_legacy_stream': ('gecko.data.streams.store', 'load_trusted_legacy_stream'), 'migrate_stream_to_v2': ('gecko.data.streams.store', 'migrate_stream_to_v2'), 'save_stream': ('gecko.data.streams.store', 'save_stream'), 'StreamBuilder': ('gecko.data.streams.builder', 'StreamBuilder'), 'StreamBundle': ('gecko.data.streams.builder', 'StreamBundle'), 'spatial_partition_slug': ('gecko.data.streams.builder', 'spatial_partition_slug'), 'spatial_profile_label': ('gecko.data.streams.builder', 'spatial_profile_label'), 'stream_identity': ('gecko.data.streams.builder', 'stream_identity')}
__all__ = ['StreamBuilder', 'StreamBundle', 'ArtifactStore', 'audit_stream', 'derive_order_variant', 'derive_order_variants', 'derive_participation_variant', 'gc_objects', 'load_stream', 'load_trusted_legacy_stream', 'migrate_stream_to_v2', 'save_stream', 'spatial_partition_slug', 'spatial_profile_label', 'stream_identity']

def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value

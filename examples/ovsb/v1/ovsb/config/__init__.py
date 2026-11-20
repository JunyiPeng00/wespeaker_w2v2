"""Recipe configuration: schema, loading, legacy import, core view.

Responsibility: turn YAML (plus overlays and command-line overrides) into a
validated, typed :class:`Config`; import a resolved config of the original
training tree; emit the wespeaker-schema dict the stock tools read.

May import: the Python standard library and ``yaml`` only (no torch, no
numpy, no other ``ovsb`` layer), so a config can be checked anywhere.

Modules: ``schema`` (dataclasses, :class:`ConfigError`), ``load`` (YAML,
overlays, overrides, derived values), ``legacy`` (``from_legacy``),
``core_view`` (``to_core_dict``).
"""

from ovsb.config.core_view import to_core_dict
from ovsb.config.legacy import from_legacy, load_legacy
from ovsb.config.load import load as load_config
from ovsb.config.load import resolve, to_dict
from ovsb.config.schema import Config, ConfigError

__all__ = ['Config', 'ConfigError', 'from_legacy', 'load_config',
           'load_legacy', 'resolve', 'to_core_dict', 'to_dict']

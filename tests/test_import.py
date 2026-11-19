"""Check the public package import against the installed distribution."""

from importlib.metadata import version

import ao_core


def test_public_version_matches_distribution() -> None:
    assert "__version__" in ao_core.__all__
    assert ao_core.__version__ == version("ao-core")

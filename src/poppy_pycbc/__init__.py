"""PyCBC interface for poppy."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("poppy-pycbc")
except PackageNotFoundError:
    __version__ = "unknown"

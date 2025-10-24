"""PyCBC interface for aspire."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aspire-pycbc")
except PackageNotFoundError:
    __version__ = "unknown"

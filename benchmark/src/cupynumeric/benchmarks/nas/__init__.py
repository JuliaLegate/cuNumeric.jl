"""NAS benchmark implementations for cuPyNumeric."""

from . import ep, ft, mg  # noqa: F401
from .ep import NASEmbarrassinglyParallel
from .ft import NASFourierTransform
from .mg import NASMultiGrid

__all__ = ["NASEmbarrassinglyParallel", "NASFourierTransform", "NASMultiGrid"]

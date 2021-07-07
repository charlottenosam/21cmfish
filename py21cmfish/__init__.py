from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .params import *
from .power_spectra import *
from .fishy import *
from .io import *
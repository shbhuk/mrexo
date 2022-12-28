"""
.. include:: ../README.md
.. include:: ../docs/dependencies.md
.. include:: ../docs/sample_usage.md

"""

from .predict import mass_100_percent_iron_planet, radius_100_percent_iron_planet
from .fit_nd import fit_relation
from .mle_utils_nd import MLE_fit
from .utils_nd import _load_lookup_table, _logging
from .cross_validate_nd import run_cross_validation
from .profile_likelihood import run_profile_likelihood
# from .aic_nd import run_aic
from .Optimizers import optimizer

__version__ = '0.3'

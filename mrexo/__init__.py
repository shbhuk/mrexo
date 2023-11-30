"""
.. include:: ../README.md
.. include:: ../docs/dependencies.md
.. include:: ../docs/sample_usage.md

"""

# from .predict import mass_100_percent_iron_planet, radius_100_percent_iron_planet
from .fit_nd import fit_relation
from .mle_utils_nd import MLE_fit, InputData
from .plotting_nd import Plot2DWeights, Plot2DJointDistribution, Plot1DInputDataHistogram
from .utils_nd import GiveDegreeCandidates
from .cross_validate_nd import RunCrossValidation
from .aic_nd import RunAIC
from .Optimizers import optimizer


__version__ = '1.1.3'


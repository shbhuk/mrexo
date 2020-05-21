"""
.. include:: ../README.md
.. include:: ../docs/dependencies.md
.. include:: ../docs/sample_usage.md

"""

from .plot import plot_y_given_x_relation, plot_x_given_y_relation, plot_yx_and_xy, plot_joint_xy_distribution, plot_mle_weights
from .predict import predict_from_measurement, mass_100_percent_iron_planet,generate_lookup_table, radius_100_percent_iron_planet
from .fit import fit_xy_relation
from .mle_utils import MLE_fit, cond_density_quantile
from .utils import _save_dictionary, _load_lookup_table, _logging
from .cross_validate import run_cross_validation

__version__ = '0.2'

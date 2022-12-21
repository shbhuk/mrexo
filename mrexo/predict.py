import numpy as np
import os
from scipy.stats.mstats import mquantiles
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from multiprocessing import Pool,cpu_count

from .mle_utils import cond_density_quantile
from .utils import _load_lookup_table
from .plot import plot_x_given_y_relation, plot_y_given_x_relation

pwd = os.path.dirname(__file__)
np.warnings.filterwarnings('ignore')



def mass_100_percent_iron_planet(logRadius):
    """
    This is from 100% iron curve of Fortney, Marley and Barnes 2007; solving for logM (base 10) via quadratic formula.
    \nINPUT:
        logRadius : Radius of the planet in log10 units
    OUTPUT:

        logMass: Mass in log10 units for a 100% iron planet of given radius
    """

    Mass_iron = (-0.4938 + np.sqrt(0.4938**2-4*0.0975*(0.7932-10**(logRadius))))/(2*0.0975)
    return Mass_iron

def radius_100_percent_iron_planet(logMass):
    """
    This is from 100% iron curve from Fortney, Marley and Barnes 2007; solving for logR (base 10) via quadratic formula.
    \nINPUT:
        logMass : Mass of the planet in log10 units
    OUTPUT:

        logRadius: Radius in log10 units for a 100% iron planet of given mass
    """

    Radius_iron = np.log10((0.0975*(logMass**2)) + (0.4938*logMass) + 0.7932)
    return Radius_iron


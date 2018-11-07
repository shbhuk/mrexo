import numpy as np
from scipy.stats import beta,norm
from scipy.integrate import quad
from scipy.optimize import brentq as root
from astropy.table import Table
import datetime,os
import sys
sys.path.append(os.path.dirname(__file__))
from MLE_fit import MLE_fit, calc_C_matrix


#' @param data.train: input training data
#' @param data.sg.train: input measurement errors for the training data
#' @param bounds: a vector contains four elements, from left to right:
#'                the upper bound for mass, the lower bound for mass
#'                the upper bound for radius, the lower bound for radius
#' @param data.test: input test data
#' @param data.sg.test: input measurement errors for the test data
#' @param deg: degree used in Bernstein polynomials
#' @param abs.tol: tolerance number of computing


def cross_validation(train_Radius, train_Mass, test_Radius, test_Mass, Mass_bounds, Radius_bounds, deg,
                        train_Radius_sigma = None, train_Mass_sigma = None,
                        test_Radius_sigma = None, test_Mass_sigma = None,
                        Log = False, abs_tol = 1e-10, location = os.path.dirname(__file__)):
    '''
    INPUT:
        VERIFY PARAMETER DESCRIPTIONS.
        train_data = Input training data
        bounds = Vecotr containing four elements. Upper and lower bound for Mass. Upper and lower bound for Radius resp.
        test_radius, = Input radius data
        test_mass, = Input mass data
        deg = Degree used in Bernstein Polynomials
        train_data_sg = Input training data measurement errors
        test_radius_sigma = Input test radius measurement errors
        test_mass_sigma = Input test mass measurement errors
        Log = Whether the data is in log units
        abs_tol = Absolute tolerance
        location : The location for the log file
    '''

    if np.shape(test_Radius_sigma) != np.shape(test_Mass_sigma):
        print('Insert error message here')
    elif np.shape(test_Radius) != np.shape(test_Mass):
        print('Gibberish')

    # Fit MLE using training dataset

    weights = MLE_fit(Mass = train_Mass, Radius = train_Radius, Mass_sigma = train_Mass_sigma, Radius_sigma = train_Radius_sigma, Mass_bounds = Mass_bounds,
            Radius_bounds = Radius_bounds, deg = deg, Log = Log,abs_tol = abs_tol, location = location, output_weights_only = True)


    size_test = np.size(test_Radius)

    # specify the bounds
    M_max = Mass_bounds[0]
    M_min = Mass_bounds[1]
    R_max = Radius_bounds[0]
    R_min = Radius_bounds[1]

    # calculate cdf and pdf of M and R for each term
    # the first and last term is set to 0 to avoid boundary effects
    # so we only need to calculate 2:(deg^2-1) terms

    C_pdf = calc_C_matrix(size_test, deg, test_Mass, test_Mass_sigma, M_max, M_min, test_Radius, test_Radius_sigma, R_max, R_min, Log, abs_tol, location)




    return np.sum(np.log(np.matmul(weights,C_pdf)))

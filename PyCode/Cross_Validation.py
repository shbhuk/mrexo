import numpy as np
from scipy.stats import beta,norm
from scipy.integrate import quad
from scipy.optimize import brentq as root
from astropy.table import Table
import datetime,os
import sys 
sys.path.append(os.path.dirname(__file__))
from MLE_fit import MLE_fit, find_indv_pdf, integrate_function
  
  
#' @param data.train: input training data
#' @param data.sg.train: input measurement errors for the training data
#' @param bounds: a vector contains four elements, from left to right:
#'                the upper bound for mass, the lower bound for mass
#'                the upper bound for radius, the lower bound for radius
#' @param data.test: input test data
#' @param data.sg.test: input measurement errors for the test data
#' @param deg: degree used in Bernstein polynomials
#' @param abs.tol: tolerance number of computing


def cross_validation(train_data, bounds, test_radius, test_mass, deg, train_data_sg = None, test_mass_sigma = None, test_radius_sigma = None, Log = False, abs_tol = 1e-10, location = os.path.dirname(__file__)):
    '''
    INPUT:
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
    
    if np.shape(test_radius_sigma) != np.shape(test_mass_sigma):
        print('Insert error message here')
    elif np.shape(test_radius) != np.shape(test_mass):
        print('Gibberish')
    
    # Fit MLE using training dataset
    weights = MLE_fit(data = train_data, bounds = bounds, sigma = train_data_sg, Log = Log, deg = deg, abs_tol = abs_tol, location = location)
    
    size_test = np.size(test_radius)
    
    # specify the bounds
    M_max = bounds[1]
    M_min = bounds[2]
    R_max = bounds[3]
    R_min = bounds[4]
    
    # calculate cdf and pdf of M and R for each term
    # the first and last term is set to 0 to avoid boundary effects
    # so we only need to calculate 2:(deg^2-1) terms
    deg_vec = np.arange(1,deg+1) 
    
    if test_radius_sigma is None:
        # PDF for Mass and Radius
        M_indv_pdf = find_indv_pdf(test_mass,deg,deg_vec,M_max,M_min) 
        R_indv_pdf = find_indv_pdf(test_radius,deg,deg_vec,R_max,R_min) 
    else:
        # PDF for Integrated beta density and normal density for Mass and Radius
        M_indv_pdf = np.array([integrate_function(data = test_mass, data_sd = test_mass_sigma, deg = deg, degree = d, x_max = M_max, x_min = M_min) for d in deg_vec])        
        R_indv_pdf = np.array([integrate_function(data = test_radius, data_sd = test_radius_sigma, deg = deg, degree = d, x_max = R_max, x_min = R_min) for d in deg_vec])  
        
        


    
    


#%cd "C:/Users/shbhu/Documents/Git/Py_mass_radius_working/PyCode"

import numpy as np
from scipy.stats import beta,norm
from scipy.integrate import quad
from scipy.optimize import brentq as root
from astropy.table import Table
from scipy.optimize import minimize, fmin_slsqp
import os
import sys

sys.path.append(os.path.dirname(__file__))
from MLE_fit import MLE_fit
    

def MR_fit(data, sigma, Mass_max = None, Mass_min = None, Radius_max = None, Radius_min = None, degree = 60, select_deg = 55,
            Log = False, k_fold = 10, num_boot = 100, bootstrap = True, store_output = False, cores = 1):
    '''
    Predict the Mass and Radius relationship
    INPUT:
                
        1) data: the first column contains the mass measurements and 
             the second column contains the radius measurements.
        2) sigma: measurement errors for the data, if no measuremnet error, 
                    it is NULL
        3) Mass_max: the upper bound for mass. Default = None
        4) Mass_min: the lower bound for mass. Default = None
        5) Radius_max: the upper bound for radius. Default = None
        6) Radius_min: the upper bound for radius. Default = None
        7) degree: the maximum degree used for cross-validation/AIC/BIC. Default = 60
        8) select_deg: if input "cv": cross validation
                            if input "aic": aic method
                            if input "bic": bic method
                            if input a number: default using that number and 
                            skip the select process 
        9) Log: is the data transformed into a log scale if Log = True. Default = False
        10) k_fold: number of fold used for cross validation. Default = 10
        11) bootstrap: if using bootstrap to obtain confidence interval. Default = True
        12) num_boot: number of bootstrap replication. Default = 100
        13) store_output: store the output into csv files if True. Default = False
        14) cores: this program uses parallel computing for bootstrap. Default = 1
      
    '''
    
    n = np.shape(data)[0]
    M = data[:,0]
    R = data[:,1]
      
    sigma_M = sigma[:,0]
    sigma_R = sigma[:,1]
    
    if len(M) != len(R):
        print('Length of Mass and Radius vectors must be the same')
    if len(M) != len(sigma_M) and (sigma_M is not None):
        print('Length of Mass and Mass sigma vectors must be the same')
    if len(R) != len(sigma_R) and (sigma_R is not None):
        print('Length of Radius and Radius sigma vectors must be the same')
        
    if Mass_min is None:
        Mass_min = np.min(M) - np.max(sigma_M)/np.sqrt(n)
    if Mass_max is None:
        Mass_max = np.max(M) + np.max(sigma_M)/np.sqrt(n)    
    if Radius_min is None:
        Radius_min = np.min(R) - np.max(sigma_R)/np.sqrt(n)
    if Radius_max is None:
        Radius_max = np.max(R) + np.max(sigma_R)/np.sqrt(n) 
        
    bounds = np.array([Mass_max,Mass_min,Radius_max,Radius_min])

    ###########################################################
    ## Step 1: Select number of degree based on cross validation, aic or bic methods.
    
    degree_candidate = np.arange(5, degree, 5)
    deg_length = len(degree_candidate)
        
    if select_deg == 'aic' : 
        aic = np.array([MLE_fit(data = data, bounds = bounds, sigma = sigma, Log = Log, deg = d)['aic'] for d in range(2,degree)])
        deg_choose = np.nanmin(aic)

    elif select_deg == 'bic':
        bic = np.array([MLE_fit(data = data, bounds = bounds, sigma = sigma, Log = Log, deg = d)['bic'] for d in range(2,degree)])
        deg_choose = np.nanmin(bic)   
                 
    elif isinstance(select_deg, (int,float)):
        deg_choose = select_deg
    
    else : 
        print('Error: Incorrect input for select_deg')
            
    ###########################################################
    ## Step 2: Estimate the model
            
            
    MLE_fit(data = data, bounds = bounds, sigma = sigma, Log = Log, deg = deg_choose)
    
    if bootstrap == True:
        
        n_boot = np.random.choice(n, n, replace = True) 
        data_boot = data[n_boot]
        data_sigma = sigma[n_boot]
        MR_boot = MLE_fit(data = data_boot, bounds = bounds, sigma = data_sigma, Log = Log, deg = deg_choose)
        
    
    
    
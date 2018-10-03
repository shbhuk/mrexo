import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import os,sys
from scipy.stats.mstats import mquantiles
pwd = os.path.dirname(__file__)
#sys.path.append(pwd)
import MLE_fit 
import importlib
importlib.reload(MLE_fit)

def predict_mass_given_radius(radius, r_sigma = None, posterior_sample = False, qtl = [0.16,0.84], islog = False):
    '''
    INPUT:
        radius = 
        
        islog = If True. Radius and Radius sigma is already in log scale. If False, will be converted to log
    
    '''
    
    # upper bounds and lower bounds used in the Bernstein polynomial model in log10 scale
    Radius_min = -0.3
    Radius_max = 1.357
    Mass_min = -1
    Mass_max = 3.809
    
    # read weights
    weights_mle = Table.read(os.path.join(pwd,'weights.mle.csv'))['#x']
    degrees = int(np.sqrt(len(weights_mle)))
    if islog == False:
        logRadius = np.log10(radius)
    else: 
        logRadius = radius
        
    
    if posterior_sample == False:
        predicted_value = MLE_fit.cond_density_quantile(y = logRadius, y_std = r_sigma, y_max = Radius_max, y_min = Radius_min,
                                                      x_max = Mass_max, x_min = Mass_min, deg = degrees, 
                                                      w_hat = weights_mle, qtl = qtl)
        predicted_mean = predicted_value[0]
        predicted_lower_quantile = predicted_value[2]
        predicted_upper_quantile = predicted_value[3]   

    elif posterior_sample == True:
        
        k = np.size(radius)
        mean_sample = np.zeros(k)
        denominator_sample = np.zeros(k)
        y_beta_indv_sample = np.zeros((k,degrees))
        
        # the model can be view as a mixture of k conditional densities for f(log m |log r), each has weight 1/k
        # the mean of this mixture density is 1/k times sum of the mean of each conditional density
        # the quantile is little bit hard to compute, and may not be avaible due to computational issues
    
        # calculate the mean
        
        for i in range(0,k):
            results = MLE_fit.cond_density_quantile(y = logRadius[i], y_std = r_sigma[i], y_max = Radius_max, y_min = Radius_min,
                                                      x_max = Mass_max, x_min = Mass_min, deg = degrees, 
                                                      w_hat = weights_mle, qtl = qtl)
                                                      
            mean_sample[i] = results[0]
            denominator_sample[i] = results[4]
            y_beta_indv_sample[i,:] = results[5:]
        
        predicted_mean = np.mean(mean_sample)

        
        
   
        
    return predicted_mean,predicted_lower_quantile,predicted_upper_quantile
            

    
    
    
    
    
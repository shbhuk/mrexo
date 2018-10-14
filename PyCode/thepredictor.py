import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import os,sys
from scipy.stats import beta

from scipy.stats.mstats import mquantiles
pwd = os.path.dirname(__file__)
#sys.path.append(pwd)
import MLE_fit 
import importlib
importlib.reload(MLE_fit)

'''
Radius = 5
Radius_sigma = 0.1
Radius_min = -0.3
Radius_max = 1.357
Mass_min = -1
Mass_max = 3.809
qtl = [0.16,0.84]
'''

weights_mle = Table.read(os.path.join(pwd,'weights.mle.csv'))['#x']

result_dir = os.path.join(pwd,'Bootstrap_open_parallel40')
#result_dir = r'C:\Users\szk381\Documents\GitHub\Py_mass_radius_working\PyCode\Results\Bootstrap_results_Apple_reduced100_bad'
#weights_mle = np.loadtxt(os.path.join(result_dir,'weights.txt'))



'''
a = MLE_fit.cond_density_quantile(y = np.log10(Radius), y_std = Radius_sigma, y_max = Radius_max, y_min = Radius_min,
                                                      x_max = Mass_max, x_min = Mass_min, deg = degrees, 
                                                      w_hat = weights_mle, qtl = qtl)

'''


def predict_mass_given_radius(Radius, Radius_sigma = None, posterior_sample = False, 
                                qtl = [0.16,0.84], islog = False,Radius_min = -0.3, 
                                Radius_max = 1.357,Mass_min = -1,Mass_max = 3.809):
    '''
    INPUT:
        Radius = The radius for which Mass is being predicted. [Earth Radii]
        Radius_sigma = 1 sigma error on radius [Earth Radii]
        posterior_sample = If the input is a posterior sample. Default is False
        qtl = Quantile values returned. Default is 0.16 and 0.84
        islog = Whether the radius given is in log scale or not. Default is False. The Radius_sigma is always in original units
        Radius, Mass = upper bounds and lower bounds used in the Bernstein polynomial model in log10 scale
    '''
    
    degrees = int(np.sqrt(len(weights_mle)))
    if islog == False:
        logRadius = np.log10(Radius)
    else: 
        logRadius = Radius
        
    
    if posterior_sample == False:
        predicted_value = MLE_fit.cond_density_quantile(y = logRadius, y_std = Radius_sigma, y_max = Radius_max, y_min = Radius_min,
                                                      x_max = Mass_max, x_min = Mass_min, deg = degrees, 
                                                      w_hat = weights_mle, qtl = qtl)
        predicted_mean = predicted_value[0]
        predicted_lower_quantile = predicted_value[2]
        predicted_upper_quantile = predicted_value[3]   

    elif posterior_sample == True:
        
        n = np.size(Radius)
        mean_sample = np.zeros(n)
        denominator_sample = np.zeros(n)
        y_beta_indv_sample = np.zeros((n,degrees))
        
        # the model can be view as a mixture of k conditional densities for f(log m |log r), each has weight 1/k
        # the mean of this mixture density is 1/k times sum of the mean of each conditional density
        # the quantile is little bit hard to compute, and may not be avaible due to computational issues
    
        # calculate the mean
        
        for i in range(0,n):
            results = MLE_fit.cond_density_quantile(y = logRadius[i], y_std = Radius_sigma[i], y_max = Radius_max, y_min = Radius_min,
                                                      x_max = Mass_max, x_min = Mass_min, deg = degrees, 
                                                      w_hat = weights_mle, qtl = qtl)
                                                      
            mean_sample[i] = results[0]
            denominator_sample[i] = results[4]
            y_beta_indv_sample[i,:] = results[5:][0]
        
        predicted_mean = np.mean(mean_sample)

        # Calculate the quantiles
        mixture_conditional_quantile = MLE_fit.mixture_conditional_density_qtl( y_max = Radius_max, y_min = Radius_min,
                                                      x_max = Mass_max, x_min = Mass_min, deg = degrees, 
                                                      w_hat = weights_mle, 
                                                      denominator_sample = denominator_sample,
                                                      y_beta_indv_sample = y_beta_indv_sample, qtl = qtl)

        
        predicted_lower_quantile = mixture_conditional_quantile[0]
        predicted_upper_quantile = mixture_conditional_quantile[1]
        
    if islog:
        return predicted_mean,predicted_lower_quantile,predicted_upper_quantile    
    else:
        return predicted_mean,predicted_lower_quantile,predicted_upper_quantile
        #return 10**predicted_mean,10**predicted_lower_quantile,10**predicted_upper_quantile
            
np.random.seed(0)
r_posterior = np.random.normal(5,0.5,10)
#print(predict_mass_given_radius(Radius = r_posterior, Radius_sigma = np.repeat(0.1,10), posterior_sample = True))  
print(predict_mass_given_radius(Radius = 5, Radius_sigma = 0.1, posterior_sample = False))  
print(predict_mass_given_radius(Radius = 5, Radius_sigma = None, posterior_sample = False))  
       
    
    
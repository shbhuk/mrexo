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

result_dir = os.path.join(pwd,'Cross_validation_20')
weights_mle = np.loadtxt(os.path.join(result_dir,'weights.txt'))



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
    print(degrees)

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

        outputs = [predicted_mean,predicted_lower_quantile,predicted_upper_quantile]

    elif posterior_sample == True:

        n = np.size(Radius)
        mean_sample = np.zeros(n)
        random_quantile = np.zeros(n)

        if len(logRadius) != len(Radius_sigma):
            print('Length of Radius array is not equal to length of Radius_sigma array. CHECK!!!!!!!')
            return 0

        for i in range(0,n):
            qtl_check = np.random.random()
            print(qtl_check)
            results = MLE_fit.cond_density_quantile(y = logRadius[i], y_std = Radius_sigma[i], y_max = Radius_max, y_min = Radius_min,
                                                      x_max = Mass_max, x_min = Mass_min, deg = degrees,
                                                      w_hat = weights_mle, qtl = [qtl_check,0.5])

            mean_sample[i] = results[0]
            random_quantile[i] = results[2]

        outputs = [mean_sample,random_quantile]

    if islog:
        return outputs
    else:
        return outputs
        #return [10**x for x in outputs]


np.random.seed(0)
r_posterior = np.random.normal(5,0.5,20)
#print(predict_mass_given_radius(Radius = r_posterior, Radius_sigma = np.repeat(0.1,20), posterior_sample = True))
print(predict_mass_given_radius(Radius = 5, Radius_sigma = 0.1, posterior_sample = False))
print(predict_mass_given_radius(Radius = 5, Radius_sigma = None, posterior_sample = False))

from mrexo import predict_from_measurement
import os
import numpy as np

try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''
    print('Could not find pwd')

'''
Sample script to show how to use the predicting function to predict mass from radius
'''

#Below example predicts the mass for a radius of 1 Earth radii exoplanet, with no measurement uncertainty from the fit results in 'M_dwarfs_dummy'
result_dir = os.path.join(pwd,'M_dwarfs_dummy')
predicted_mass, qtls = predict_from_measurement(measurement=1, measurement_sigma=None, result_dir=result_dir, is_posterior=False)

#Below example predicts the mass for a radius of 1 Earth radii exoplanet with uncertainty of 0.1 Earth Radii on the included Mdwarf fit.
#Similary for Kepler dataset. Also outputs 16,84% qtl
predicted_mass, qtls = predict_from_measurement(measurement=1, measurement_sigma=0.1, result_dir=None, dataset='mdwarf', is_posterior=False)

#Below example predicts the mass for a radius of 1 Earth radii exoplanet with uncertainty of 0.1 Earth Radii on the included Mdwarf fit.
#Similary for Kepler dataset. Also output 5,16,84,95% quantile
predicted_mass, qtls = predict_from_measurement(measurement=1, measurement_sigma=0.1, result_dir=None, dataset='mdwarf', is_posterior=False,
                       qtl = [0.05,0.16,0.84,0.95])


'''
Sample script to show how to use the predicting function to predict radius from mass
'''


#Below example predicts the radius for a mass of 1 Earth mass exoplanet with uncertainty of 0.1 Earth Mass on the included Mdwarf fit. Similary for Kepler dataset.
predicted_mass, qtls = predict_from_measurement(measurement=1, measurement_sigma=0.1, predict = 'radius', result_dir=None, dataset='mdwarf', is_posterior=False)

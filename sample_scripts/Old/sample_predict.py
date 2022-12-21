from mrexo import predict_from_measurement, generate_lookup_table
import os
import numpy as np
import matplotlib.pyplot as plt


try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''
    print('Could not find pwd')

'''
Sample script to show how to use the predicting function to predict mass from radius
'''


# Predict mass and quantiles from radius for a 1 Earth radii planet with an uncertainty of 0.1 radii using the M dwarf fit from the result_dir
result_dir = os.path.join(pwd,'M_dwarfs_dummy')
predicted_mass, qtls, iron_planet = predict_from_measurement(measurement=1, measurement_sigma=0.1, result_dir=result_dir, is_posterior=False, show_plot=False)
print(predicted_mass)

# Predict mass from radius for the Kepler dataset for a 1 Earth radii planet
predicted_mass, qtls, iron_planet = predict_from_measurement(measurement=1, measurement_sigma=None, predict = 'mass', dataset='kepler')

#Predict the mass measurement from a dummy radius posterior and plot it
posterior, iron_planet = predict_from_measurement(measurement=np.random.normal(1,0.1,1000),
            measurement_sigma=None, result_dir=None, dataset='mdwarf', is_posterior=True, show_plot=True, use_lookup = True)

# Predict the mass for a radius of 1 Earth radii exoplanet with uncertainty of 0.1 Earth Radii on the included Mdwarf fit.
# Also output 5,16,84,95% quantile
predicted_mass, qtls, iron_planet = predict_from_measurement(measurement=1, measurement_sigma=0.1, result_dir=None, dataset='mdwarf', is_posterior=False,
                       qtl = [0.05,0.16,0.84,0.95], show_plot=False)
                       
# OR 
predicted_mass, qtls, iron_planet = predict_from_measurement(measurement=1, measurement_sigma=0.1,qtl = [0.05,0.16,0.84,0.95])

'''
Sample script to show how to use the predicting function to predict radius from mass
'''

#Below example predicts the radius for a mass of 1 Earth mass exoplanet with uncertainty of 0.1 Earth Mass on the included Mdwarf fit. Similary for Kepler dataset.
predicted_radius, qtls, iron_planet = predict_from_measurement(measurement=1, measurement_sigma=0.1, predict = 'radius', result_dir=None, dataset='mdwarf', is_posterior=False)

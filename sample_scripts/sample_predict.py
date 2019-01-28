from mrexo import predict_from_measurement
import os
import numpy as np
pwd = '~/mrexo_working/'

#Below example predicts the mass for a radius of log10(1) Earth radii exoplanet, with no measurement uncertainty from the fit results in 'M_dwarfs_deg_cv'
result_dir = os.path.join(pwd,'M_dwarfs_deg_cv')
predicted_mass, qtls = predict_from_measurement(measurement=1, measurement_sigma=None, result_dir=result_dir, is_posterior=False, is_log=True)

#Below example predicts the mass for a radius of log10(1) Earth radii exoplanet with uncertainty of 0.1 Earth Radii on the included Mdwarf fit. Similary for Kepler dataset.
predicted_mass, qtls = predict_from_measurement(measurement=1, measurement_sigma=0.1, result_dir=None, dataset='mdwarf', is_posterior=False, is_log=True)

#Below example predicts the radius for a mass of log10(1) Earth mass exoplanet with uncertainty of 0.1 Earth Mass on the included Mdwarf fit. Similary for Kepler dataset.
predicted_mass, qtls = predict_from_measurement(measurement=1, measurement_sigma=0.1, predict = 'radius', result_dir=None, dataset='mdwarf', is_posterior=False, is_log=True)    

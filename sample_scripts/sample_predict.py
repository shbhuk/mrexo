from mrexo import predict_from_measurement, generate_lookup_table
import os
import numpy as np
import datetime
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d


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

# predicted_mass, qtls = predict_from_measurement(measurement=1, measurement_sigma=None, result_dir=result_dir, use_lookup = True)


# result_dir = r'C:\Users\shbhu\Documents\GitHub\mrexo\mrexo\datasets\M_dwarfs_20181214'
# result_dir = os.path.join(pwd,'M_dwarfs_new_17')

# 
# predict_quantity = 'Mass'
# cores = 5
# 
# 
# predict_quantity = predict_quantity.replace(' ', '').replace('-', '').lower()
# 
# input_location = os.path.join(result_dir, 'input')
# output_location = os.path.join(result_dir, 'output')
# Mass_min, Mass_max = np.loadtxt(os.path.join(input_location, 'Mass_bounds.txt'))
# Radius_min, Radius_max = np.loadtxt(os.path.join(input_location, 'Radius_bounds.txt'))
# 
# lookup_grid_size = 20
# 
# lookup_table = np.zeros((lookup_grid_size, lookup_grid_size))
# qtl_steps = np.linspace(0,1,lookup_grid_size)
# qtl_steps[-1] = 1 - 1e-20
# 
# 
# if predict_quantity == 'mass':
#     search_steps = np.linspace(Radius_min, Radius_max, lookup_grid_size)
#     fname = 'lookup_m_given_r'
#     comment = 'Lookup table for predicting log(Mass) given log(Radius) and certain quantile.'
# else:
#     search_steps = np.linspace(Mass_min, Mass_max, lookup_grid_size)
#     fname = 'lookup_r_given_m'
#     comment = 'Lookup table for predicting log(Radius) given log(Mass) and certain quantile.'
# 
# def partial_predict(measurement):
#     return predict_from_measurement(measurement = measurement, qtl = qtl_steps,
#                                     result_dir = result_dir, predict = predict_quantity)[1]
# print(1)
# 
# if cores<=1:
#     for i in range(0,lookup_grid_size):
#         print(datetime.datetime.now())
#         lookup_table[i,:] = partial_predict(measurement = search_steps[i])
#         
#         if i%100==0:
#             print(i)
# else:
#     if __name__ == '__main__':
#         print('parallel')
#         pool = Pool(processes=cores)
#         lookup_table = list(pool.map(partial_predict,search_steps))
#         print(np.shape(lookup_table))
#         plt.imshow(lookup_table)
#         plt.show()

# np.savetxt(os.path.join(output_location,fname+'.txt'), lookup_table, comments='#', header=comment)

# interp = interp2d(qtl_steps, search_steps, lookup_table)
# np.save(os.path.join(output_location,fname+'_interp2d.npy'), interp)

if __name__ == '__main__':
    #generate_lookup_table(result_dir = result_dir, predict_quantity = 'mass', cores = 24)
    #generate_lookup_table(result_dir = result_dir, predict_quantity = 'radius', cores = 24)
    result_dir = os.path.join(pwd,'Kepler_55_new_pdf')
    a=1
    generate_lookup_table(result_dir = result_dir, predict= 'mass', cores = 24)
    generate_lookup_table(result_dir = result_dir, predict_quantity = 'radius', cores = 24)

    print(predict_from_measurement(measurement=1, measurement_sigma=None, result_dir=result_dir, use_lookup = True))

# print(predicted_mass, qtls)
"""

#Below example predicts the mass for a radius of 1 Earth radii exoplanet with uncertainty of 0.1 Earth Radii on the included Mdwarf fit.
#Similary for Kepler dataset. Also outputs 16,84% qtl
predicted_mass, qtls = predict_from_measurement(measurement=1, measurement_sigma=0.1, result_dir=None, dataset='mdwarf', is_posterior=False, show_plot=False)

#Below example predicts the mass for a radius of 1 Earth radii exoplanet with uncertainty of 0.1 Earth Radii on the included Mdwarf fit.
#Similary for Kepler dataset. Also output 5,16,84,95% quantile
predicted_mass, qtls = predict_from_measurement(measurement=1, measurement_sigma=0.1, result_dir=None, dataset='mdwarf', is_posterior=False,
                       qtl = [0.05,0.16,0.84,0.95], show_plot=True)


'''
Sample script to show how to use the predicting function to predict radius from mass
'''


#Below example predicts the radius for a mass of 1 Earth mass exoplanet with uncertainty of 0.1 Earth Mass on the included Mdwarf fit. Similary for Kepler dataset.
predicted_mass, qtls = predict_from_measurement(measurement=1, measurement_sigma=0.1, predict = 'radius', result_dir=None, dataset='mdwarf', is_posterior=False)


"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import os,sys

pwd = os.path.dirname(__file__)
#sys.path.append(pwd)
import MLE_fit
from thepredictor import predict_mass_given_radius


degrees = np.arange(5,35)
directories = ['M_dwarfs_degree_{}'.format(x) for x in degrees]

results = np.zeros((3, len(degrees)))

for i,d in enumerate(directories):
    result_dir = os.path.join(pwd, d)
    weights_mle = np.loadtxt(os.path.join(result_dir,'weights.txt'))
    prediction = predict_mass_given_radius(Radius = 0, posterior_sample = False, islog = True, weights_mle = weights_mle, Radius_max = 1.304, Mass_max = 2.4357, Mass_min = -2)
    results[0,i] = degrees[i]
    results[1,i] = prediction[0]
    results[2,i] = np.mean([prediction[0] - prediction[1], prediction[2] - prediction[0]] )

    #print(degrees[i], prediction)

#plt.plot(results[0,:], np.log10(results[1,:]), '.')
plt.errorbar(results[0,:], results[1,:], yerr = results[2,:], fmt = 'o')
plt.xlabel('No of degrees used')
plt.ylabel('log Mass predicted')
plt.title('M-dwarf mass radius prediction for input log(R) = 0')
plt.show()

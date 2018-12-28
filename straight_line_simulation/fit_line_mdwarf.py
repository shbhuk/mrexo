import os
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import cpu_count

from mrexo import fit_mr_relation
from scipy.odr import Model, RealData, ODR


pwd = os.path.dirname(__file__)

#pwd = '~/mrexo_working/'

t = Table.read(os.path.join(pwd,'Cool_stars_MR_20181214_exc_upperlim.csv'))

# Symmetrical errorbars
Mass_sigma = (abs(t['pl_masseerr1']) + abs(t['pl_masseerr2']))/2
Radius_sigma = (abs(t['pl_radeerr1']) + abs(t['pl_radeerr2']))/2

# In Earth units
Mass = np.array(t['pl_masse'])
Radius = np.array(t['pl_rade'])

logMass = np.log10(Mass)
logRadius = np.log10(Radius)
logMass_sigma = 0.434 * Mass_sigma/Mass
logRadius_sigma = 0.434 * Radius_sigma/Radius

p = np.poly1d(np.polyfit(logRadius, logMass, 1))





def f(B, x):
    '''Linear function y = m*x + b'''
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return B[0]*x + B[1]

linear = Model(f)
data = RealData(logRadius, logMass, logMass_sigma, logRadius_sigma)

# Set up ODR with the model and data.
odr = ODR(data, linear, beta0=[0., 1.])

# Run the regression.
out = odr.run()
slope = out.beta[0]
intercept = out.beta[1]


'''
plt.plot(logRadius,logMass, '.')
plt.show()


'''



R_min = np.min(Radius)*1
R_max = np.max(Radius)*1


sim_sizes = [20,50,100]
#sim_sizes = [10]
intrinsic_disp = [0.,0.1,0.5,1.0]
#intrinsic_disp = [1.0]


for i in sim_sizes:
    for j in intrinsic_disp:
        data_size = i

        log_sim_radius_init = np.linspace(np.log10(R_min),np.log10(R_max), data_size)
        log_sim_mass_init = slope*log_sim_radius_init + intercept

        lin_sim_radius_error = 10**log_sim_radius_init * 0.1
        lin_sim_radius = 10**log_sim_radius_init + np.random.normal(0, np.abs(lin_sim_radius_error))

        # lin_sim_mass_intrinsic = 10**log_sim_mass_init + np.random.normal(0, np.abs(lin_sim_mass_error))
        log_sim_mass_intrinsic = log_sim_mass_init + np.random.normal(0, j, data_size)

        lin_sim_mass_error = 10**log_sim_mass_intrinsic * 0.1

        log_sim_mass = np.log10(10**log_sim_mass_intrinsic + np.random.normal(0, np.abs(lin_sim_mass_error)))


        # Directory to store results in
        result_dir = os.path.join(pwd)

        print('Simulation_{}pts_{}disp'.format(data_size, j))

        #print(min(10**sim_radius - 10**sim_radius_error))

        # plt.errorbar(x=10**log_sim_radius_init, y=10**log_sim_mass_init, xerr=lin_sim_radius_error, yerr=lin_sim_mass_error, fmt='k.',markersize=2, elinewidth=0.3)
        # plt.errorbar(x=lin_sim_radius, y=lin_sim_mass_intrinsic, xerr=lin_sim_radius_error, yerr=lin_sim_mass_error, fmt='k.',markersize=2, elinewidth=0.3)
        # plt.errorbar(x=lin_sim_radius, y=10**log_sim_mass, xerr=lin_sim_radius_error, yerr=lin_sim_mass_error, fmt='k.',markersize=2, elinewidth=0.3)

        # plt.plot(10**sim_radius_init, 10**sim_mass_init, '.')
        # plt.errorbar(x=10**sim_radius, y=10**sim_mass, xerr=10**sim_radius_error, yerr=10**sim_mass_error, fmt='k.',markersize=2, elinewidth=0.3)


        if __name__ == '__main__':
            initialfit_result, bootstrap_results = fit_mr_relation(Mass=10**log_sim_mass, Mass_sigma = lin_sim_mass_error,
                                                    Radius = lin_sim_radius, Radius_sigma = lin_sim_radius_error,
                                                    save_path = os.path.join(result_dir,'Simulation_{}pts_{}disp'.format(data_size, j)), select_deg = 'cv',
                                                    num_boot = cpu_count(), cores = cpu_count())

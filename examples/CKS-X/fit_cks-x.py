import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse

# sys.path.append("/Users/hematthi/Documents/MRExo/mrexo/")
from mrexo.mle_utils_nd import InputData, MLE_fit, calc_C_matrix
from mrexo.Optimizers import optimizer, LogLikelihood
from mrexo.fit_nd import fit_relation
from mrexo.plotting_nd import Plot2DJointDistribution, Plot2DWeights, Plot1DInputDataHistogram


# pwd = os.path.dirname(__file__)



##### To set up the data and inputs:

# To read the CSV data file:
table = pd.read_csv(r'C:\Users\skanodia\Documents\GitHub\mrexo\examples\CKS-X\CKS-X_planets_stars.csv')

table = table[(table['Rp'] < 10.) & (table['E_Mstar-iso'] != 0)] # filter out very large planet radii, including some spurious values (a few planets have thousands of Earth radii)

# To select the dimensions to model:
# (Should choose one of either period or bolometric flux)
periods = np.array(table['Per']) # orbital periods (days)
#periods_uerr = np.array(table['E_Per']) # period upper errors
#periods_lerr = np.array(abs(table['e_Per'])) # period lower errors
periods_uerr = np.empty(len(periods))
periods_uerr[:] = np.nan # set period errors to NaNs for now because they are too small
periods_lerr = periods_uerr

bolfluxes = np.array(table['S']) # incident bolometric fluxes ('Sgeo' units?)
bolfluxes_uerr = np.array(table['E_S'])
bolfluxes_lerr = np.array(abs(table['e_S']))

radii = np.array(table['Rp']) # planet radii (Earth radii)
radii_uerr = np.array(table['E_Rp'])
radii_lerr = np.array(abs(table['e_Rp']))

stmasses = np.array(table['Mstar-iso']) # stellar masses (Solar masses)
stmasses_uerr = np.array(table['E_Mstar-iso'])
stmasses_lerr = np.array(abs(table['e_Mstar-iso']))

# To set the bounds for each dimension:
print('Min/max periods: [%s, %s] days' % (np.min(periods), np.max(periods)))
print('Min/max fluxes: [%s, %s] Sgeo' % (np.min(bolfluxes), np.max(bolfluxes)))
print('Min/max planet radii: [%s, %s] R_earth' % (np.min(radii), np.max(radii)))
print('Min/max stellar masses: [%s, %s] M_sun' % (np.min(stmasses), np.max(stmasses)))

period_bounds = [0.1, 550.]
bolflux_bounds = [0.1, 6000.]
radius_bounds = [0.4, 10.]
stmass_bounds = [0.4, 1.6]

# To construct the data dictionaries:
period_dict = {'Data': periods, 'LSigma': periods_lerr, 'USigma': periods_uerr, 'Max': np.log10(period_bounds[1]), 'Min': np.log10(period_bounds[0]), 'Label': 'Period (days)', 'Char': 'P'}
bolflux_dict = {'Data': bolfluxes, 'LSigma': bolfluxes_lerr, 'USigma': bolfluxes_uerr, 'Max': np.log10(bolflux_bounds[1]), 'Min': np.log10(bolflux_bounds[0]), 'Label': 'Bolometric flux (Sgeo)', 'Char': 'S'}
radius_dict = {'Data': radii, 'LSigma': radii_lerr, 'USigma': radii_uerr, 'Max': np.log10(radius_bounds[1]), 'Min': np.log10(radius_bounds[0]), 'Label': 'Planet radius ($R_\oplus$)', 'Char': 'Rp'}
stmass_dict = {'Data': stmasses, 'LSigma': stmasses_lerr, 'USigma': stmasses_uerr, 'Max': np.log10(stmass_bounds[1]), 'Min': np.log10(stmass_bounds[0]), 'Label': 'Stellar mass ($M_\odot$)', 'Char': 'Mstar'}

input_dicts = [bolflux_dict, radius_dict] # period_dict, bolflux_dict
# input_dicts = [radius_dict, period_dict]
DataDict = InputData(input_dicts)
ndim = len(input_dicts)

select_deg = [20, 20]

run_name = 'CKS-X_insol_radius_d60'
save_path = os.path.join(run_name)



##### To run the model fitting:

if __name__ == '__main__':
	# No Monte Carlo drawing of parameters or bootstrap sampling of data:
	outputs = fit_relation(DataDict, select_deg='aic', save_path=save_path, degree_max=30, cores=2, SymmetricDegreePerDimension=True, NumMonteCarlo=0, NumBootstrap=0)

	_ = Plot1DInputDataHistogram(save_path)

	if ndim==2:
			
		_ = Plot2DJointDistribution(save_path)
		_ = Plot2DWeights(save_path)

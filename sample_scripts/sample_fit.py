import os
from astropy.table import Table
import numpy as np
from multiprocessing import cpu_count
import numpy as np


from mrexo import fit_xy_relation
from mrexo import predict_from_measurement
import pandas as pd

try :
	pwd = os.path.dirname(__file__)
except NameError:
	pwd = 'C:\\\\Users\\\\shbhu\\\\Documents\\\\GitHub\\\\mrexo\\\\sample_scripts'
	print('Could not find pwd')


DataDirectory = r'C:\Users\shbhu\Documents\GitHub\Mdwarf-Exploration\Data\MdwarfPlanets'
t = pd.read_csv(os.path.join(DataDirectory, 'Teff_4000_ExcUpperLimits_20210115.csv'))
t = pd.read_csv(os.path.join(DataDirectory, 'Teff_4000_IncUpperLimits_20210115_Metallicity.csv'))

# t = Table.read(os.path.join(pwd,'Kepler_MR_inputs.csv'))
# t = Table.read(os.path.join(pwd,'FGK_20190406.csv'))

# Symmetrical errorbars
Mass_sigma1 = np.array(abs(t['pl_masseerr1']))
Mass_sigma2 = np.array(abs(t['pl_masseerr1']))
Radius_sigma1 = np.array(abs(t['pl_radeerr1']))
Radius_sigma2 = np.array(abs(t['pl_radeerr1']))

# In Earth units
Mass = np.array(t['pl_masse'])
Radius = np.array(t['pl_rade'])
Period = np.array(t['pl_orbper'])
Period_sigma = np.repeat(np.nan, len(Period))

StellarMass = np.array(t['st_mass'])
StellarMass_Sigma1 = np.array(t['st_masserr1'])

Metallicity = np.array(t['st_metfe'])
MetallicitySigma = np.array(t['st_metfeerr1'])

# Directory to store results in
result_dir = os.path.join(pwd,'Mdwarfs_20200520_cv50')
# result_dir = os.path.join(pwd,'Kepler127_aic')
# result_dir = os.path.join(pwd, 'FGK_319_cv100')

# Run with 100 bootstraps. Selecting degrees to be 17. Alternatively can set select_deg = 'cv' to
# find the optimum number of degrees.

FakePeriod = np.ones(len(Period))
# FakePeriodSigma = FakePeriod*0.01
# Period_sigma = FakePeriodSigma
"""
# Simulation
Radius = 10**np.linspace(0, 1)
Radius_sigma1 = np.repeat(np.nan, len(Radius))
Radius_sigma2 = np.repeat(np.nan, len(Radius))
Mass = 10**(2*np.log10(Radius)*np.log10(Radius) - 0.5*np.log10(Radius))
# Mass =  np.ones(len(Radius))/2
Mass_sigma1 = np.repeat(np.nan, len(Radius))
Mass_sigma2 = np.repeat(np.nan, len(Radius))
Period = np.ones(len(Radius))/2
StellarMass_Sigma1 = np.repeat(np.nan, len(Radius))
Period = Radius# np.linspace(0, 1, len(Radius))
Period_sigma = np.repeat(np.nan, len(Radius))
"""

RadiusDict = {'Data': Radius, 'SigmaLower': Radius_sigma1,  "SigmaUpper":Radius_sigma2, 'Max':np.nan, 'Min':np.nan, 'Label':'Radius', 'Char':'r'}
MassDict = {'Data': Mass, 'SigmaLower': Mass_sigma1, "SigmaUpper":Mass_sigma2, 'Max':np.nan, 'Min':np.nan, 'Label':'Mass', 'Char':'m'}
FakePeriodDict = {'Data': FakePeriod, 'SigmaLower': FakePeriodSigma, "SigmaUpper":FakePeriodSigma, 'Max':np.nan, 'Min':np.nan, 'Label':'Period', 'Char':'p'}
PeriodDict = {'Data': Period, 'SigmaLower': Period_sigma, "SigmaUpper":Period_sigma, 'Max':np.nan, 'Min':np.nan, 'Label':'Period', 'Char':'p'}
StellarMassDict = {'Data': StellarMass, 'SigmaLower': StellarMass_Sigma1, "SigmaUpper":StellarMass_Sigma1, 'Max':np.nan, 'Min':np.nan, 'Label':'StellarMass', 'Char':'stm'}
MetallicityDict = {'Data': Metallicity, 'SigmaLower': MetallicitySigma, "SigmaUpper":MetallicitySigma, 'Max':np.nan, 'Min':np.nan, 'Label':'Metallicity', 'Char':'feh'}

from mrexo.mle_utils_nd import InputData, MLE_fit, _find_indv_pdf
from mrexo.fit_nd import fit_relation
import matplotlib.pyplot as plt
InputDictionaries = [MassDict, RadiusDict, PeriodDict]
InputDictionaries = [RadiusDict, StellarMassDict, PeriodDict, MetallicityDict]
# InputDictionaries = [MassDict, RadiusDict, FakePeriodDict]
# InputDictionaries = [MassDict, RadiusDict]
DataDict = InputData(InputDictionaries)
save_path = 'C:\\Users\\shbhu\\Documents\\GitHub\\mrexo\\sample_scripts\\Trial_nd'
 
ndim = len(InputDictionaries)
deg_per_dim = [25, 25, 25, 30]
deg_per_dim = [20] * ndim
"""
outputs = MLE_fit(DataDict, 
	deg_per_dim=deg_per_dim,
	save_path=save_path, OutputWeightsOnly=False, CalculateJointDist=True)
"""

outputs, _ = fit_relation(DataDict, select_deg=deg_per_dim, save_path=save_path, num_boot=0)

JointDist = outputs['JointDist']
weights = outputs['Weights']

"""
################ Plot Joint Distribution ################ 
plt.figure()
plt.imshow(JointDist[:,:,5].T, 
	extent=(DataDict['ndim_bounds'][0][0], DataDict['ndim_bounds'][0][1], DataDict['ndim_bounds'][1][0], DataDict['ndim_bounds'][1][1]), 
	aspect='auto', origin='lower'); 
plt.plot(np.log10(DataDict['ndim_data'][0]), np.log10(DataDict['ndim_data'][1]), 'k.')
# plt.title("Orbital Period = {} d".format(str(np.round(title,3))))
plt.ylabel("Log10 "+DataDict['ndim_label'][1]);
plt.xlabel("Log10 "+DataDict['ndim_label'][0]);
plt.tight_layout()
plt.show(block=False)

plt.imshow(weights.reshape(deg_per_dim))
plt.show(block=False)

# x = MassDict
y = StellarMassDict

c = MassDict
x = RadiusDict
# y = PeriodDict

x = RadiusDict
y = MassDict
c = MetallicityDict

# plt.scatter(x['Data'], y['Data'], c=c['Data'])
plt.scatter(np.log10(x['Data']), np.log10(y['Data']), c=np.log10(c['Data']))
plt.colorbar(label=c['Label'])
plt.xlabel(x['Label'])
plt.ylabel(y['Label'])
plt.tight_layout()
plt.show(block=False)

# plt.imshow(JointDist[:,:,50], extent=(x.min(), x.max(), y.min(), y.max()), aspect='auto', origin='lower');
# plt.xlabel("log10 Mass")
# plt.ylabel("log10 Radius")

################ Run Conditional Distribution ################ 
from mrexo.mle_utils_nd import calculate_conditional_distribution1D, calculate_conditional_distribution2D

ConditionString = 'm|r,p'
# ConditionString = 'm|r'
# ConditionString = 'm,r|p'
ConditionString = 'm,r|stm'
# ConditionString = 'm|r,stm'
# ConditionString = 'm,r|p,stm'

DataDict = DataDict
MeasurementDict = {'r':[[1, 2], [np.nan, np.nan]], 'p':[[1, 1], [np.nan, np.nan]], 'stm':[[0.1, 0.3], [np.nan, np.nan]]}
MeasurementDict = {'r':[[10**0.2, 10**0.4, 10**0.6], [np.nan, np.nan, np.nan]]}#, 'p':[[1, 1, 10], [np.nan, np.nan]], 'stm':[[0.5], [np.nan, np.nan]]}
MeasurementDict = {'stm':[[0.2, 0.4, 0.43, 0.46, 0.49, 0.52, 0.55, 0.57, 0.6], [np.nan]*9]}#, 'r':[[1], [np.nan]]}
# MeasurementDict = {'r':[[1, 1, 1], [np.nan, np.nan, np.nan]], 'p':[[1, 5, 10], [np.nan, np.nan, np.nan]]}
LogMeasurementDict = {k:np.log10(MeasurementDict[k]) for k in MeasurementDict.keys()}



ConditionalDist, MeanPDF, VariancePDF = calculate_conditional_distribution(ConditionString, DataDict, weights, deg_per_dim,
	JointDist, LogMeasurementDict)



x = outputs['DataSequence'][0]
y = outputs['DataSequence'][1]
z = outputs['DataSequence'][2]
# t = outputs['DataSequence'][3]

i=8
plt.imshow(ConditionalDist[i], extent=(x.min(), x.max(), y.min(), y.max()), aspect='auto', origin='lower'); 
plt.plot(np.log10(Mass), np.log10(Radius),  'k.')
plt.title(DataDict['ndim_label'][2]+" = " + str(MeasurementDict['stm'][0][i]))
plt.ylabel("Log10 Radius");
plt.xlabel("Log10 Mass");
plt.tight_layout()
plt.show(block=False)



"""
"""
from mrexo.mle_utils import calculate_joint_distribution

joint = calculate_joint_distribution(X_points=x, X_min=x.min(), X_max=x.max(), 
	Y_points=y, Y_min=y.min(), Y_max=y.max(), weights=weights, abs_tol=1e-8)

i = 10




from mrexo.mle_utils_nd import cond_density_quantile
# a = radius
# b = mass 
# when m|r

mean, var, quantile, denominator, a_beta_indv = cond_density_quantile(a=np.log10(1),
	a_min=DataDict['ndim_bounds'][1][0], a_max=DataDict['ndim_bounds'][1][1], 
	b_max=DataDict['ndim_bounds'][0][1], b_min=DataDict['ndim_bounds'][0][0], 
	deg=deg_per_dim[0], deg_vec=np.arange(1,deg_per_dim[0]+1), w_hat=weights)
"""

"""
if __name__ == '__main__':
			initialfit_result, _ = fit_xy_relation(**RadiusDict, **MassDict,
												save_path = result_dir, select_deg = 'aic',
												num_boot = 100, cores = 4, degree_max=100)

"""

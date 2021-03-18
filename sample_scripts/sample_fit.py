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

UTeff = 6200

t = pd.read_csv(os.path.join(DataDirectory, 'Teff_{}_ExcUpperLimits_20210316.csv'.format(UTeff)))
t = pd.read_csv(os.path.join(DataDirectory, 'Teff_{}_ExcUpperLimits_20210316_Metallicity.csv'.format(UTeff)))
t = pd.read_csv(os.path.join(DataDirectory, 'Teff_{}_IncUpperLimits_20210316_Metallicity.csv'.format(UTeff)))
t = pd.read_csv(os.path.join(DataDirectory, 'Teff_{}_IncUpperLimits_20210316.csv'.format(UTeff)))



# t = pd.read_csv(os.path.join(DataDirectory, 'Teff_4400_IncUpperLimits_20210127.csv'))
# t = pd.read_csv(os.path.join(DataDirectory, 'Teff_4400_IncUpperLimits_20210127_Metallicity.csv'))
# t = pd.read_csv(os.path.join(DataDirectory, 'Teff_6000_ExcUpperLimits_20210216.csv'))

t = t[t['st_mass'] < 10]
# t= t[t['pl_hostname'] != 'TRAPPIST-1']
# t = t[t['pl_masse'] < 50]
print(len(t))
# t = t[np.isfinite(t['pl_insol'])]
# t = t[t['pl_insol'] > 1e-10]

# t = Table.read(os.path.join(pwd,'Kepler_MR_inputs.csv'))
# t = Table.read(os.path.join(pwd,'FGK_20190406.csv'))


# In Earth units
Mass = np.array(t['pl_masse'])
# Symmetrical errorbars
Mass_sigma1 = np.array(abs(t['pl_masseerr1']))
Mass_sigma2 = np.array(abs(t['pl_masseerr1']))

Radius = np.array(t['pl_rade'])
# Symmetrical errorbars
Radius_sigma1 = np.array(abs(t['pl_radeerr1']))
Radius_sigma2 = np.array(abs(t['pl_radeerr1']))

Period = np.array(t['pl_orbper'])
PeriodSigma = np.repeat(np.nan, len(Period))

StellarMass = np.array(t['st_mass'])
StellarMassSigma = np.array(t['st_masserr1'])

Metallicity = np.array(t['st_metfe'])
MetallicitySigma = np.array(t['st_metfeerr1'])

Insolation = np.array(t['pl_insol'])
InsolationSigma = np.array(t['pl_insolerr1'])



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
Radius_sigma1 = 0.1*Radius# np.repeat(np.nan, len(Radius))
Radius_sigma2 = np.copy(Radius_sigma1)
Mass = 10**(2*np.log10(Radius)*np.log10(Radius) - 0.5*np.log10(Radius))
# Mass =  np.ones(len(Radius))/2
Mass_sigma1 = np.repeat(np.nan, len(Radius))
Mass_sigma2 = np.repeat(np.nan, len(Radius))
Period = np.ones(len(Radius))/2
StellarMass_Sigma1 = np.repeat(np.nan, len(Radius))
Period = Radius# np.linspace(0, 1, len(Radius))
Period_sigma = 0.1*Period #np.repeat(np.nan, len(Radius))
# """

Max, Min = 1, 0
Max, Min = np.nan, np.nan
# Max += 0.2
# Min -= 0.2

RadiusDict = {'Data': Radius, 'SigmaLower': Radius_sigma1,  "SigmaUpper":Radius_sigma2, 'Max':Max, 'Min':Min, 'Label':'Radius ($R_{\oplus}$)', 'Char':'r'}
MassDict = {'Data': Mass, 'SigmaLower': Mass_sigma1, "SigmaUpper":Mass_sigma2, 'Max':Max, 'Min':np.nan, 'Label':'Mass ($M_{\oplus}$)', 'Char':'m'}
FakePeriodDict = {'Data': FakePeriod, 'SigmaLower': PeriodSigma, "SigmaUpper":PeriodSigma, 'Max':np.nan, 'Min':np.nan, 'Label':'Period (d)', 'Char':'p'}
PeriodDict = {'Data': Period, 'SigmaLower': PeriodSigma, "SigmaUpper":PeriodSigma, 'Max':Max, 'Min':Min, 'Label':'Period (d)', 'Char':'p'}
StellarMassDict = {'Data': StellarMass, 'SigmaLower': StellarMassSigma, "SigmaUpper":StellarMassSigma, 'Max':np.nan, 'Min':np.nan, 'Label':'Stellar Mass (M$_{\odot}$)', 'Char':'stm'}
MetallicityDict = {'Data': 10**Metallicity, 'SigmaLower': np.repeat(np.nan, len(Metallicity)), "SigmaUpper":np.repeat(np.nan, len(Metallicity)), 'Max':np.nan, 'Min':np.nan, 'Label':'Metallicity [Fe/H]', 'Char':'feh'}
# MetallicityDict = {'Data': 10**Metallicity, 'SigmaLower': np.repeat(np.nan, len(Metallicity)), "SigmaUpper":np.repeat(np.nan, len(Metallicity)), 'Max':1, 'Min':-0.45, 'Label':'Metallicity [Fe/H]', 'Char':'feh'}
# MetallicityDict = {'Data': 10**(-Metallicity), 'SigmaLower': np.repeat(np.nan, len(Metallicity)), "SigmaUpper":np.repeat(np.nan, len(Metallicity)), 'Max':np.nan, 'Min':np.nan, 'Label':'Metallicity [Fe/H]', 'Char':'feh'}

InsolationDict = {'Data': Insolation, 'SigmaLower': InsolationSigma, "SigmaUpper":InsolationSigma, 'Max':np.nan, 'Min':np.nan, 'Label':'Pl Insol ($S_{\oplus}$)', 'Char':'insol'}

from mrexo.mle_utils_nd import InputData, MLE_fit, _find_indv_pdf
from mrexo.fit_nd import fit_relation
import matplotlib.pyplot as plt
InputDictionaries = [RadiusDict, MassDict, PeriodDict]
InputDictionaries = [RadiusDict, MassDict, StellarMassDict]
# InputDictionaries = [RadiusDict, MassDict, FakePeriodDict]
# InputDictionaries = [RadiusDict, MassDict,  MetallicityDict]

# InputDictionaries = [RadiusDict, StellarMassDict, PeriodDict, MetallicityDict]
# InputDictionaries = [RadiusDict, StellarMassDict, PeriodDict]
# InputDictionaries = [RadiusDict, MassDict, ]
# InputDictionaries = [RadiusDict, PeriodDict, StellarMassDict]
InputDictionaries = [RadiusDict, PeriodDict, StellarMassDict]
InputDictionaries = [RadiusDict, PeriodDict, MetallicityDict]

# InputDictionaries = [RadiusDict, InsolationDict, StellarMassDict]
# InputDictionaries = [RadiusDict, StellarMassDict]
DataDict = InputData(InputDictionaries)
save_path = 'C:\\Users\\shbhu\\Documents\\GitHub\\mrexo\\sample_scripts\\Trial_nd'
 
ndim = len(InputDictionaries)
deg_per_dim = [25, 25, 25, 30]
deg_per_dim = [35] * ndim
# deg_per_dim = [25, 26]
"""
outputs = MLE_fit(DataDict, 
	deg_per_dim=deg_per_dim,
	save_path=save_path, OutputWeightsOnly=False, CalculateJointDist=True)
"""

outputs, _ = fit_relation(DataDict, select_deg=deg_per_dim, save_path=save_path, num_boot=0)

JointDist = outputs['JointDist']
weights = outputs['Weights']


plt.figure()
size = int(np.sqrt(len(weights)))

plt.imshow(np.reshape(weights , [size, size]), extent = [0, size, 0, size], origin = 'left')
plt.xticks(np.arange(0,size), *[np.arange(0,size)])
plt.yticks(np.arange(0,size), *[np.arange(0,size)])
plt.title(deg_per_dim)
# plt.imshow(weights.reshape(deg_per_dim))
plt.colorbar()
plt.show(block=False)


################ Plot Joint Distribution ################ 
x = DataDict['DataSequence'][0]
y = DataDict['DataSequence'][1]

fig = plt.figure(figsize=(8.5,6.5))
im = plt.imshow(JointDist, 
	extent=(DataDict['ndim_bounds'][0][0], DataDict['ndim_bounds'][0][1], DataDict['ndim_bounds'][1][0], DataDict['ndim_bounds'][1][1]), 
	aspect='auto', origin='lower'); 
plt.errorbar(x=np.log10(DataDict['ndim_data'][0]), y=np.log10(DataDict['ndim_data'][1]), xerr=0.434*DataDict['ndim_sigma'][0]/DataDict['ndim_data'][0], yerr=0.434*DataDict['ndim_sigma'][1]/DataDict['ndim_data'][1], fmt='.', color='k', alpha=0.4)
# plt.title("Orbital Period = {} d".format(str(np.round(title,3))))
plt.ylabel("Log10 "+DataDict['ndim_label'][1]);
plt.xlabel("Log10 "+DataDict['ndim_label'][0]);
# plt.xlabel("Planetary Mass ($M_{\oplus}$)")
# plt.ylabel("Planetary Radius ($R_{\oplus}$)")
plt.xlim(DataDict['ndim_bounds'][0][0], DataDict['ndim_bounds'][0][1])
plt.ylim(DataDict['ndim_bounds'][1][0], DataDict['ndim_bounds'][1][1])
plt.tight_layout()
XTicks = np.linspace(x.min(), x.max(), 5)
# XTicks = np.log10(np.array([0.3, 1, 3, 10, 30, 100, 300]))
YTicks = np.linspace(y.min(), y.max(), 5)
# YTicks = np.log10(np.array([1, 3, 5, 10]))


XLabels = np.round(10**XTicks, 1)

YLabels = np.round(10**YTicks, 2)

plt.xticks(XTicks, XLabels)
plt.yticks(YTicks, YLabels)
cbar = fig.colorbar(im, ticks=[np.min(JointDist), np.max(JointDist)], fraction=0.037, pad=0.04)
cbar.ax.set_yticklabels(['Min', 'Max'])
plt.tight_layout()
plt.show(block=False)


# plt.imshow(weights.reshape(deg_per_dim))
# plt.show(block=False)

# x = MassDict
y = StellarMassDict

c = StellarMassDict
x = RadiusDict
y = MassDict

x = RadiusDict
y = StellarMassDict
c= PeriodDict

# plt.scatter(x['Data'], y['Data'], c=np.log10(c['Data']))
plt.scatter(x['Data'], y['Data'], c=c['Data'])
# plt.scatter(np.log10(x['Data']), np.log10(y['Data']), c=np.log10(c['Data']))
plt.colorbar(label=c['Label'])
plt.xlabel(x['Label'])
plt.ylabel(y['Label'])
plt.tight_layout()
plt.show(block=False)

# plt.imshow(JointDist[:,:,50], extent=(x.min(), x.max(), y.min(), y.max()), aspect='auto', origin='lower');
# plt.xlabel("log10 Mass")
# plt.ylabel("log10 Radius")

################ Run Conditional Distribution ################ 
from mrexo.mle_utils_nd import calculate_conditional_distribution

ConditionString = 'm|r,p'
# ConditionString = 'm|r'
# ConditionString = 'm,r|p'
ConditionString = 'r,p|stm'
# ConditionString = 'm|r,stm'
# ConditionString = 'm,r|p,stm'
# ConditionString = 'm,r|feh'
# ConditionString = 'm,r|p'
# ConditionString = 'r|stm'


DataDict = DataDict
MeasurementDict = {'r':[[1, 2], [np.nan, np.nan]], 'p':[[1, 1], [np.nan, np.nan]], 'stm':[[0.1, 0.3], [np.nan, np.nan]]}
MeasurementDict = {'r':[[10**0.2, 10**0.4, 10**0.6], [np.nan, np.nan, np.nan]]}#, 'p':[[1, 1, 10], [np.nan, np.nan]], 'stm':[[0.5], [np.nan, np.nan]]}
MeasurementDict = {'stm':[[0.2, 0.4, 0.43, 0.46, 0.49, 0.52, 0.55, 0.57, 0.6], [np.nan]*9]}#, 'r':[[1], [np.nan]]}
# MeasurementDict = {'r':[[1, 1, 1], [np.nan, np.nan, np.nan]], 'p':[[1, 5, 10], [np.nan, np.nan, np.nan]]}
MeasurementDict = {'feh':[[10**0.0], [np.nan]]}
# MeasurementDict = {'r':[[1], [np.nan]]}

MeasurementDict = {'stm':[[0.2], [np.nan]]}
LogMeasurementDict = {k:np.log10(MeasurementDict[k]) for k in MeasurementDict.keys()}

# 20210310
# So for a 2D case, need to pass transpose of Joint Dist 

ConditionalDist, MeanPDF, VariancePDF = calculate_conditional_distribution(ConditionString, DataDict, weights, deg_per_dim,
	JointDist, LogMeasurementDict)

# plt.plot(x, ConditionalDist[0], label=MeasurementDict['stm'][0])


x = outputs['DataSequence'][0]
y = outputs['DataSequence'][1]
z = outputs['DataSequence'][2]
# t = outputs['DataSequence'][3]

i=0
plt.imshow(ConditionalDist[i], extent=(x.min(), x.max(), y.min(), y.max()), aspect='auto', origin='lower'); 
# plt.plot(np.log10(Mass), np.log10(Radius),  'k.')
# plt.title(DataDict['ndim_label'][2]+" = " + str(np.log10(MeasurementDict['feh'][0][i])))
# plt.xlabel(x['Label'])
# plt.ylabel(y['Label'])
plt.tight_layout()
plt.show(block=False)

_ = NumericalIntegrate2D(x, y, ConditionalDist[0], [x.min(), x.max()], [y.min(), y.max()])
print(_)

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

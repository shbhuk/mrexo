import os, sys
from astropy.table import Table
import numpy as np
from multiprocessing import cpu_count
import numpy as np


from mrexo import fit_xy_relation
from mrexo import predict_from_measurement
import pandas as pd

Platform = sys.platform

if Platform == 'win32':
	HomeDir =  'C:\\Users\\shbhu\\Documents\\\\GitHub\\'
else:
	HomeDir = r"/storage/home/szk381/work/"


try :
	pwd = os.path.dirname(__file__)
except NameError:
	pwd = os.path.join(HomeDir, 'mrexo', 'sample_scripts')
	print('Could not find pwd')


NPoints = 50

Radius = np.linspace(0, 1, NPoints)
RadiusSigma = 0.2*Radius
RadiusSigma = 0
Radius = 10**(Radius + np.random.normal(0, RadiusSigma))
Radius = np.random.lognormal(0, 1, NPoints)
RadiusSigma = np.repeat(None, len(Radius))
Radius_sigma1 = RadiusSigma
Radius_sigma2 = np.copy(Radius_sigma1)

# Quadratic Mass
Mass = 10**(2*np.log10(Radius)*np.log10(Radius) - 0.5*np.log10(Radius))

# Constant Mass
# Mass =  np.ones(len(Radius))/2

Mass = np.concatenate([np.random.lognormal(0, 1, NPoints//2), \
	np.random.lognormal(5, 1, NPoints//2)])

Mass_sigma1 = np.repeat(np.nan, len(Radius))
Mass_sigma2 = np.repeat(np.nan, len(Radius))

# Constant Period
Period = np.ones(len(Radius))/2

# Period = Radius
Period = Radius# np.linspace(0, 1, len(Radius))
PeriodSigma = 0.2*Period #np.repeat(np.nan, len(Radius))
PeriodSigma = np.repeat(np.nan, len(Radius))


Max, Min = np.nan, np.nan

RadiusDict = {'Data': Radius, 'SigmaLower': Radius_sigma1,  "SigmaUpper":Radius_sigma2, 'Max':Max, 'Min':Min, 'Label':'Radius ($R_{\oplus}$)', 'Char':'r'}
MassDict = {'Data': Mass, 'SigmaLower': Mass_sigma1, "SigmaUpper":Mass_sigma2, 'Max':Max, 'Min':np.nan, 'Label':'Mass ($M_{\oplus}$)', 'Char':'m'}
PeriodDict = {'Data': Period, 'SigmaLower': PeriodSigma, "SigmaUpper":PeriodSigma, 'Max':Max, 'Min':Min, 'Label':'Period (d)', 'Char':'p'}


from mrexo.mle_utils_nd import InputData, MLE_fit
from mrexo.fit_nd import fit_relation
import matplotlib.pyplot as plt
InputDictionaries = [RadiusDict, MassDict, PeriodDict]
InputDictionaries = [RadiusDict, MassDict]


DataDict = InputData(InputDictionaries)
save_path = os.path.join(pwd, 'Trial_Sim_2D_{}'.format(NPoints))

ndim = len(InputDictionaries)
deg_per_dim = [25, 25, 25, 30]
deg_per_dim = [100] * ndim
# deg_per_dim = [35, 30]

outputs, _ = fit_relation(DataDict, select_deg=deg_per_dim, save_path=save_path, num_boot=0)

JointDist = outputs['JointDist']
weights = outputs['Weights']
unpadded_weight = outputs['UnpaddedWeights']

"""
################# PLOT WEIGHTS ######################
plt.figure()
plt.imshow(np.reshape(weights , deg_per_dim).T, origin = 'left')
# plt.xticks(np.arange(0,size), *[np.arange(0,size)])
# plt.yticks(np.arange(0,size), *[np.arange(0,size)])
plt.title(deg_per_dim)
plt.colorbar()
plt.show(block=False)

plt.figure()
plt.imshow(np.reshape(unpadded_weight, np.array(deg_per_dim)-2).T, origin='lower')
plt.show(block=False)
"""

################ Plot Joint Distribution ################ 
x = DataDict['DataSequence'][0]
y = DataDict['DataSequence'][1]

fig = plt.figure(figsize=(8.5,6.5))
im = plt.imshow(JointDist.T, 
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



####################
from scipy.interpolate import RectBivariateSpline

x = DataDict['DataSequence'][0]
y = DataDict['DataSequence'][1]

z = JointDist 
z /= z.max()

interp = RectBivariateSpline(x, y, z)


def Draw2D(x, y, interp):
	
	xrand = np.random.uniform(x.min(), x.max())
	yrand = np.random.uniform(y.min(), y.max())

	Probability = interp(xrand, yrand)

	u = np.random.random()
	steps = 0
	
	while u > Probability:
		xrand = np.random.uniform(x.min(), x.max())
		yrand = np.random.uniform(y.min(), y.max())

		Probability = interp(xrand, yrand)

		u = np.random.random()
		steps +=1 
		
		if steps > 500:
			continue
	return np.array([xrand, yrand])
	
	
RandomDraw = np.array([Draw2D(x, y, interp) for i in range(50)])

plt.imshow(z, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower')
plt.plot(RandomDraw[:,0], RandomDraw[:,1], '.')
plt.show(block=False)

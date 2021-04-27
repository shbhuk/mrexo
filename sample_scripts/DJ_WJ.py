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

df = pd.read_csv(os.path.join(pwd, 'Dong2021_Table4.csv'))

st_rad = np.array(df.rstar)
st_raderr1 = np.array(df.rstar_upper)
st_raderr2 = np.array(df.rstar_lower)

st_rho = np.array(df.rhostar)
st_rhoerr1 = np.array(df.rhostar_upper)
st_rhoerr2 = np.array(df.rhostar_lower)

pl_orbper = np.array(df.period)
pl_orbpererr1 = np.repeat(np.nan, len(pl_orbper)) # np.array(df.period_upper)
pl_orbpererr2 = np.repeat(np.nan, len(pl_orbper)) # np.array(df.period_lower)

eccen = np.array(df.e)
eccenerr1 = np.repeat(np.nan, len(eccen)) #np.array(df.e_upper)
eccenerr2 = np.repeat(np.nan, len(eccen)) # np.array(df.e_lower)

Max, Min = np.nan, np.nan
RadiusDict = {'Data': st_rad, 'SigmaLower': st_raderr2,  "SigmaUpper":st_raderr1, 'Max':Max, 'Min':Min, 'Label':'St Radius ($R_{\odot}$)', 'Char':'r'}
DensityDict = {'Data': st_rho, 'SigmaLower': st_rhoerr2,  "SigmaUpper":st_rhoerr1, 'Max':Max, 'Min':Min, 'Label':'St Density', 'Char':'rho'}
PeriodDict = {'Data': pl_orbper, 'SigmaLower': pl_orbpererr2,  "SigmaUpper":pl_orbpererr1, 'Max':Max, 'Min':Min, 'Label':'Period (d)', 'Char':'p'}
EccenDict = {'Data': eccen, 'SigmaLower': eccenerr2,  "SigmaUpper":eccenerr1, 'Max':Max, 'Min':Min, 'Label': 'Eccentricity', 'Char':'e'}


from mrexo.mle_utils_nd import InputData, MLE_fit, _find_indv_pdf
from mrexo.fit_nd import fit_relation
import matplotlib.pyplot as plt

InputDictionaries = [RadiusDict, PeriodDict, EccenDict]
# InputDictionaries = [PeriodDict, EccenDict]

DataDict = InputData(InputDictionaries)
save_path = os.path.join(pwd, 'TestRuns', 'DJ_RadiusPeriodEccen')
ndim = len(InputDictionaries)


# outputs, _ = fit_relation(DataDict, select_deg='aic', save_path=save_path, num_boot=0, degree_max=25)
outputs, _ = fit_relation(DataDict, select_deg=[25, 25,25], save_path=save_path, num_boot=0, degree_max=25)


JointDist = outputs['JointDist']
weights = outputs['Weights']
unpadded_weight = outputs['UnpaddedWeights']
deg_per_dim = outputs['deg_per_dim']

"""

plt.figure()
plt.imshow(np.reshape(weights , deg_per_dim).T, origin = 'left', aspect='auto')
plt.title(deg_per_dim)
plt.colorbar()
plt.savefig(os.path.join(save_path, 'output', 'weights.png'))
plt.close('all')

################ Plot Joint Distribution ################ 
x = DataDict['DataSequence'][0]
y = DataDict['DataSequence'][1]

fig = plt.figure(figsize=(8.5,6.5))
im = plt.imshow(JointDist.T, 
	extent=(DataDict['ndim_bounds'][0][0], DataDict['ndim_bounds'][0][1], DataDict['ndim_bounds'][1][0], DataDict['ndim_bounds'][1][1]), 
	aspect='auto', origin='lower'); 
plt.errorbar(x=np.log10(DataDict['ndim_data'][0]), y=np.log10(DataDict['ndim_data'][1]), xerr=0.434*DataDict['ndim_sigma'][0]/DataDict['ndim_data'][0], yerr=0.434*DataDict['ndim_sigma'][1]/DataDict['ndim_data'][1], fmt='.', color='k', alpha=0.4)
plt.ylabel("Log10 "+DataDict['ndim_label'][1]);
plt.xlabel("Log10 "+DataDict['ndim_label'][0]);
plt.xlim(DataDict['ndim_bounds'][0][0], DataDict['ndim_bounds'][0][1])
plt.ylim(DataDict['ndim_bounds'][1][0], DataDict['ndim_bounds'][1][1])
plt.tight_layout()

XTicks = np.linspace(x.min(), x.max(), 5)
YTicks = np.linspace(y.min(), y.max(), 5)
XLabels = np.round(10**XTicks, 1)
YLabels = np.round(10**YTicks, 2)
plt.xticks(XTicks, XLabels)
plt.yticks(YTicks, YLabels)

cbar = fig.colorbar(im, ticks=[np.min(JointDist), np.max(JointDist)], fraction=0.037, pad=0.04)
cbar.ax.set_yticklabels(['Min', 'Max'])
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'output', 'JointDist.png'))
plt.close("all")
# plt.show(block=False)

"""

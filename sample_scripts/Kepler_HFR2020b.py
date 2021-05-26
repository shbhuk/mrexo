import os, sys
#from astropy.table import Table
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

Dataset = 'Real'
#Dataset = 'Simulated'

if Dataset == 'Real':
	df = pd.read_csv(os.path.join(pwd, 'q1_q17_dr25_gaia_berger_fgk_HFR2020b_koi_cleaned.csv'))
	# df = df[df['koi_disposition'] == 'CONFIRMED']
else:
	df = pd.read_csv(os.path.join(pwd, 'observed_catalog.csv'))
	df = df.loc[:len(df)//5 - 1]


RadiusBounds = None
RadiusBounds = [0, 4]

if RadiusBounds is not None:
	df = df[df['koi_prad'] > RadiusBounds[0]]
	df = df[df['koi_prad'] < RadiusBounds[1]]

pl_orbper = np.array(df.koi_period)
pl_orbpererr1 = np.repeat(np.nan, len(pl_orbper)) # np.array(df.period_upper)
pl_orbpererr2 = np.repeat(np.nan, len(pl_orbper)) # np.array(df.period_lower)

pl_rade = np.array(df.koi_prad)
pl_radeerr1 = np.repeat(np.nan, len(pl_rade)) #np.array(df.e_upper)
pl_radeerr2 = np.repeat(np.nan, len(pl_rade)) # np.array(df.e_lower)

st_rad = np.array(df.koi_srad)
# st_raderr1 = np.array(df.rstar_upper)
# st_raderr2 = np.array(df.rstar_lower)
st_raderr1 = np.repeat(np.nan, len(st_rad))
st_raderr2 = np.repeat(np.nan, len(st_rad))

st_mass = np.array(df.koi_smass)
st_masserr1 = np.repeat(np.nan, len(st_mass))
st_masserr2 = np.repeat(np.nan, len(st_mass))

# st_teff = np.array(df.koi_steff)
# st_tefferr1 = np.repeat(np.nan, len(st_teff))
# st_tefferr2 = np.repeat(np.nan, len(st_teff))


Max, Min = np.nan, np.nan

PeriodDict = {'Data': pl_orbper, 'SigmaLower': pl_orbpererr2,  "SigmaUpper":pl_orbpererr1, 'Max':Max, 'Min':Min, 'Label':'Period (d)', 'Char':'p'}
RadiusDict = {'Data': pl_rade, 'SigmaLower': pl_radeerr1,  "SigmaUpper":pl_radeerr2, 'Max':Max, 'Min':Min, 'Label':'Radius ($R_{\oplus}$)', 'Char':'r'}

StellarMassDict = {'Data': st_mass, 'SigmaLower': st_masserr1, "SigmaUpper":st_masserr1, 'Max':np.nan, 'Min':np.nan, 'Label':'Stellar Mass (M$_{\odot}$)', 'Char':'stm'}
StellarRadiusDict = {'Data': st_rad, 'SigmaLower': st_raderr2,  "SigmaUpper":st_raderr1, 'Max':Max, 'Min':Min, 'Label':'St Radius ($R_{\odot}$)', 'Char':'str'}
# TeffDict = {'Data': st_teff, 'SigmaLower': st_tefferr1, "SigmaUpper":st_tefferr2, 'Max':np.nan, 'Min':np.nan, 'Label':'Stellar Teff (K)', 'Char':'teff'}


from mrexo.mle_utils_nd import InputData, MLE_fit, _find_indv_pdf
from mrexo.fit_nd import fit_relation
import matplotlib.pyplot as plt

if Dataset == 'Real':
	RunName = 'Kepler'
else:
	RunName = 'SimKepler'

RunName = RunName + '_HFR2020b_'
RunName = RunName + 'RP_deg1000'

if RadiusBounds is not None:
	RunName = RunName + '_'+str(RadiusBounds[0])+'_'+str(RadiusBounds[1])
	
print(RunName)
print("========================")
InputDictionaries = [PeriodDict, RadiusDict]
#InputDictionaries = [PeriodDict, RadiusDict, StellarMassDict]
# InputDictionaries = [StellarMassDict, RadiusDict]

# InputDictionaries = [PeriodDict, RadiusDict, StellarMassDict]


DataDict = InputData(InputDictionaries)
save_path = os.path.join(pwd, 'TestRuns', RunName)
ndim = len(InputDictionaries)


outputs, _ = fit_relation(DataDict, select_deg='aic', save_path=save_path, num_boot=0, degree_max=1000, cores=6)
#outputs, _ = fit_relation(DataDict, select_deg=[25, 45], save_path=save_path, num_boot=0, degree_max=25)


JointDist = outputs['JointDist']
weights = outputs['Weights']
unpadded_weight = outputs['UnpaddedWeights']
deg_per_dim = outputs['deg_per_dim']

for n in range(ndim):
	x = DataDict['ndim_data'][n]
	plt.hist(x, bins=np.logspace(np.log10(x.min()), np.log10(x.max()), 20+1))
	plt.xlabel(DataDict['ndim_label'][n])
	plt.gca().set_xscale("log")
	plt.title(RunName)
	plt.tight_layout()
	plt.savefig(os.path.join(save_path, 'output', 'Histogram_'+DataDict['ndim_char'][n]+'.png'))
	plt.close("all")



if ndim==2:
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
	plt.ylabel(DataDict['ndim_label'][1]);
	plt.xlabel(DataDict['ndim_label'][0]);
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



import os, sys
from astropy.table import Table
import numpy as np
from multiprocessing import cpu_count
import numpy as np

import pandas as pd

from mrexo.mle_utils_nd import InputData, MLE_fit,InvertHalfNormalPDF
from mrexo.fit_nd import fit_relation
from mrexo.plotting_nd import Plot2DJointDistribution, Plot2DWeights, Plot1DInputDataHistogram
import matplotlib.pyplot as plt

Platform = sys.platform

if Platform == 'win32':
	HomeDir =  'C:\\Users\\skanodia\\Documents\\\\GitHub\\'
else:
	HomeDir = r"/storage/home/szk381/work/"
	#HomeDir = r"/home/skanodia/work/"


try :
	pwd = os.path.dirname(__file__)
except NameError:
	pwd = os.path.join(HomeDir, 'mrexo', 'sample_scripts')
	print('Could not find pwd')


DataDirectory = os.path.join(HomeDir, 'Mdwarf-Exploration', 'Data', 'Class2Disks', 'Ansdell2016_Lupus')
print(DataDirectory)

t = pd.read_csv(os.path.join(DataDirectory, 'Combined_t1t2t3_noNans.csv'))

stmass = np.asarray(t['Mass'])
stmasserr1 = np.asarray(t['e_Mass'])

diskdustmass = np.asarray(t['MDust'])
diskdustmasserr1 = np.asarray(t['e_MDust'])
diskdustmasserr2 = np.asarray(t['e_MDust']).copy()

UpperLimits = np.where(t['MDust'] < 0)
DustUpperLimits = (diskdustmass + 3*diskdustmasserr1)[UpperLimits]

diskdustmass[UpperLimits] = DustUpperLimits
diskdustmasserr1[UpperLimits] = InvertHalfNormalPDF(x=DustUpperLimits, p=0.997)
diskdustmasserr2[UpperLimits] = 0


print(len(t))



Max, Min = 1, 0
Max, Min = np.nan, np.nan

StellarMassDict = {'Data': stmass, 'LSigma': stmasserr1, "USigma":stmasserr1, 'Max':Max, 'Min':Min, 'Label':'Stellar Mass (M$_{\odot}$)', 'Char':'stm'}
DiskDustMassDict = {'Data': diskdustmass, 'LSigma': diskdustmasserr1, "USigma":diskdustmasserr2, 'Max':Max, 'Min':Min, 'Label':'Disk Dust Mass (M$_{\oplus}$)', 'Char':'mdust'}

InputDictionaries = [StellarMassDict, DiskDustMassDict]
DataDict = InputData(InputDictionaries)

ndim = len(InputDictionaries)


for d in [20]:#, 40, 80, 100, 500, 1000]:
	# print(d)

	RunName = 'LupusClassII_2d_CV_100MC_100BS'
	#RunName = 'AllPlanet_RpLt20_Sparse_MR'
	save_path = os.path.join(pwd, 'TestRuns',  RunName)

	deg_per_dim = [25, 25, 25]

	select_deg = 'cv'

	if __name__ == '__main__':

		outputs= fit_relation(DataDict, select_deg=select_deg, save_path=save_path, degree_max=50, cores=6,SymmetricDegreePerDimension=True, NumMonteCarlo=100, NumBootstrap=100)

		_ = Plot1DInputDataHistogram(save_path)

		if ndim==2:
				
			_ = Plot2DJointDistribution(save_path)
			_ = Plot2DWeights(save_path)


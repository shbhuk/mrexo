a = r"""

a = 3
a_std = 1
a_LSigma = 1
a_USigma = 1

start = datetime.datetime.now()
for i in range(1000):
	_ComputeOldConvolvedPDF(a, deg, deg_vec, a_max, a_min, a_std=a_std, abs_tol=1e-8, Log=False)
end = datetime.datetime.now()
print(end-start)

start = datetime.datetime.now()
for i in range(1000):
	_ComputeConvolvedPDF(a, deg, deg_vec, a_max, a_min, 
		a_LSigma=a_LSigma, a_USigma=a_USigma,
		abs_tol=1e-8, Log=False)
end = datetime.datetime.now()
print(end-start)

i = 50
dim = 1

a=DataDict["ndim_data"][dim][i]
a_LSigma=DataDict["ndim_LSigma"][dim][i]
a_USigma=DataDict["ndim_USigma"][dim][i]
deg=deg_per_dim[dim]
deg_vec=deg_vec_per_dim[dim]
a_max=DataDict["ndim_bounds"][dim][1]
a_min=DataDict["ndim_bounds"][dim][0]

'''
start = datetime.datetime.now()
for i in range(10000):
	_ = stats.norm.pdf(i, i, 100)
end = datetime.datetime.now()
print(end-start)

start = datetime.datetime.now()
for i in range(1000000):
	_ = _PDF_Normal(i, i, 100)
end = datetime.datetime.now()
print(end-start)

n = 103

start = datetime.datetime.now()
for i in range(1000000):
	_ = _GammaFunction(n)
end = datetime.datetime.now()
print(end-start)

from math import factorial, gamma

start = datetime.datetime.now()
for i in range(1000000):
	_ = factorial(n-1)
end = datetime.datetime.now()
print(end-start)

start = datetime.datetime.now()
for i in range(1000000):
	_ = scipy.math.factorial(n-1)
end = datetime.datetime.now()
print(end-start)
'''

R_points = np.array(pd.read_csv(r"C:\Users\skanodia\Downloads\Compare_MRExo\Test_N100_Diagonal_0.1Error_Scatter\R_points.csv").iloc[:,1])
M_points = np.array(pd.read_csv(r"C:\Users\skanodia\Downloads\Compare_MRExo\Test_N100_Diagonal_0.1Error_Scatter\M_points.csv").iloc[:,1])

qtls = np.arange(0, 100)

# ECDF for mass given R=6 
M_cond_R6 = pd.read_csv(r"C:\Users\skanodia\Downloads\Compare_MRExo\Test_N100_Diagonal_0.1Error_Scatter\M_cond_R6_qtl100.csv")
R6_cdf = np.array(M_cond_R6.iloc[:,1])

InterpCDF = interp1d(qtls, R6_cdf, bounds_error=False, fill_value=(R6_cdf[0], R6_cdf[-1]))

RSample = []

for i in range(100000):
	p = np.random.uniform(0, 1)*100
	RSample.append(InterpCDF(p))

plt.hist(RSample, density=True)

####################################################

from mrexo.mle_utils_nd import calc_C_matrix, _ComputeConvolvedPDF
from mrexo.Optimizers import optimizer, LogLikelihood
from mrexo.utils_nd import _logging
import datetime
from scipy import sparse

'''
deg_per_dim = [120, 120]

s = datetime.datetime.now()
C_pdf = calc_C_matrix(DataDict, deg_per_dim, abs_tol=1e-8, save_path='', verbose=2, UseSparseMatrix=UseSparseMatrix)
w = optimizer(C_pdf, deg_per_dim, verbose=2, save_path='', MaxIter=500, rtol=1e-3, UseSparseMatrix=UseSparseMatrix)
e = datetime.datetime.now()

print("Using Sparse Matrix = "+str(UseSparseMatrix))
print("C_pdf shape =", np.shape(C_pdf), ':: NumElements x 1e6 = ', np.prod(np.shape(C_pdf))/1e6)
if UseSparseMatrix:
	print("Size for C_pdf in MB = {:.3f}, LogLikelihood = {:.1f}, 95% weight = {}".format(C_pdf.data.size/(1024**2), w[1], np.percentile(w[0], q=95)))
	# print("{:.2f}% of elements are < 1e-10".format(100*np.size(C_pdf.toarray()[C_pdf.toarray() < 1e-10])/np.size(C_pdf.toarray())))
else:
	print("Size for C_pdf in MB = {:.3f}, LogLikelihood = {:.1f}, 95% weight = {}".format(C_pdf.nbytes/(1024**2), w[1], np.percentile(w[0], q=95)))
	# print("{:.2f}% of elements are < 1e-10".format(100*np.size(C_pdf[C_pdf < 1e-10])/np.size(C_pdf)))
print(e-s)
"""

from multiprocessing import Pool
from mrexo.mle_utils_nd import MLE_fit, _ComputeConvolvedPDF
from mrexo.cross_validate_nd import run_cross_validation
# from .profile_likelihood import run_profile_likelihood
from mrexo.utils_nd import _logging, _save_dictionary, GiveDegreeCandidates
from mrexo.aic_nd import run_aic, _AIC_MLE

import numpy as np
from scipy.stats import beta,norm
import scipy
from scipy.stats import beta

from decimal import Decimal
from scipy.stats import rv_continuous
from scipy.integrate import quad
from scipy.optimize import brentq as root
from scipy.interpolate import interpn, interp1d	
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import RectBivariateSpline
from scipy.special import erfinv
from scipy import sparse

import datetime,os
from multiprocessing import current_process
from functools import lru_cache
import tracemalloc
from mrexo.Optimizers import optimizer, LogLikelihood

import os, sys
from astropy.table import Table
import numpy as np
from multiprocessing import cpu_count
import numpy as np

import pandas as pd

import cProfile
import pstats


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


from memory_profiler import profile
import tracemalloc
@profile

def calc_C_matrix(DataDict, deg_per_dim,
		abs_tol, save_path, verbose, SaveCMatrix=False,
		UseSparseMatrix=False):
	'''
	Integrate the product of the normal and beta distributions for Y and X and then take the Kronecker product.
	2D matrix with shape = (N x product(degrees-2)). For example in two dimensions this would be (N x (d1-2).(d2-2))

	Refer to Ning et al. 2018 Sec 2.2 Eq 8 and 9.
	'''


	message = 'Starting C matrix calculation at {}'.format(datetime.datetime.now())
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

	ndim = DataDict['ndim']
	n = DataDict['DataLength']
	# For degree 'd', actually using d-2 since the boundaries are zero padded.
	deg_vec_per_dim = [np.arange(1, deg-1) for deg in deg_per_dim] 
	indv_pdf_per_dim = [np.zeros((n, deg-2)) for deg in deg_per_dim]
	
	# Product of degrees (deg-2 since zero padded)
	deg_product = 1
	for deg in deg_per_dim:
		deg_product *= deg-2
		
	# tracemalloc.start()
	if UseSparseMatrix:
		C_pdf = sparse.lil_matrix((deg_product, n))
		message = 'Using Sparse Matrix for C_pdf'
		_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)
	else:
		C_pdf = np.zeros((deg_product, n))

	# Loop across each data point.
	for i in range(0,n):
		# if UseSparseMatrix:
			# kron_temp = sparse.csr_matrix(1)
		# else:
		kron_temp = 1
		for dim in range(0,ndim):
			indv_pdf_per_dim[dim][i,:] = _ComputeConvolvedPDF(a=DataDict["ndim_data"][dim][i], 
				a_LSigma=DataDict["ndim_LSigma"][dim][i], a_USigma=DataDict["ndim_USigma"][dim][i],
				deg=deg_per_dim[dim], deg_vec=deg_vec_per_dim[dim],
				a_max=DataDict["ndim_bounds"][dim][1], 
				a_min=DataDict["ndim_bounds"][dim][0], 
				Log=True)
			
			# kron_temp = np.kron(indv_pdf_per_dim[dim][i,:], kron_temp) # Old method
			
			# Starting 20210323, we're flipping the order for the kron product
			# because there seems to be a flipping of degrees, only apparent in the asymmetric degree case

			# if UseSparseMatrix:
				# kron_temp = sparse.kron(kron_temp, indv_pdf_per_dim[dim][i,:])
			# else:
			kron_temp = np.kron(kron_temp, indv_pdf_per_dim[dim][i,:])
			kron_temp[kron_temp <= 1e-10] = 0

		# if UseSparseMatrix:
			# C_pdf[:,i] = kron_temp
		# else:
		C_pdf[:,i] = kron_temp
	
	# print(tracemalloc.get_traced_memory())
	# tracemalloc.stop()
	# print("Sparsity = ", 1 - np.count_nonzero(C_pdf.todense())/C_pdf.todense().size)

	message = 'Finished Integration at {}. \nCalculated the PDFs for Integrated beta and normal density.'.format(datetime.datetime.now())
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

	if SaveCMatrix:
		np.savetxt(os.path.join(save_path, 'C_pdf.txt'), C_pdf.toarray())
	return C_pdf


DataDirectory = os.path.join(HomeDir, 'Mdwarf-Exploration', 'Data', 'MdwarfPlanets')
print(DataDirectory)

t = pd.read_csv(os.path.join(DataDirectory, 'Teff_7000_ExcUpperLimits_20230306RpLt20.csv'))
# t = pd.read_csv(os.path.join(DataDirectory, 'Teff_7000_ExcUpperLimits_20220401_Thesis.csv'))
# t = pd.read_csv(os.path.join(pwd, 'Cool_stars_20181214_exc_upperlim.csv'))
# t = pd.read_csv(os.path.join(pwd, 'Kepler_MR_inputs.csv'))

t = t[~np.isnan(t['pl_insolerr1'])]
t = t[~np.isnan(t['pl_masse'])]


RadiusBounds = [0, 10]# None# [0, 100]
MassBounds = None# [0, 6000]
InsolationBounds = None# [0.01, 5000]
StellarMassBounds = None# [0.2, 1.2]

t['st_masserr1'][t['st_masserr1'] < 0.005] = np.nan
t['st_masserr2'][t['st_masserr2'] < 0.005] = np.nan

if RadiusBounds is not None:
	t = t[(t.pl_rade > RadiusBounds[0]) & (t.pl_rade < RadiusBounds[1])]

if MassBounds is not None:
	t = t[(t.pl_masse > MassBounds[0]) & (t.pl_masse < MassBounds[1])]

if InsolationBounds is not None:
	t = t[(t.pl_insol > InsolationBounds[0]) & (t.pl_insol < InsolationBounds[1])]
	
if StellarMassBounds is not None:
	t = t[(t.st_mass > StellarMassBounds[0]) & (t.st_mass < StellarMassBounds[1])]
	

RemovePlanets = ['Kepler-54 b', 'Kepler-54 c']
t = t[~np.isin(t.pl_name, RemovePlanets)]


print(len(t))

# In Earth units
Mass = np.array(t['pl_masse'])
# Symmetrical errorbars
MassUSigma = np.array(abs(t['pl_masseerr1']))
MassLSigma = np.array(abs(t['pl_masseerr2']))

Radius = np.array(t['pl_rade'])
# Symmetrical errorbars
RadiusUSigma = np.array(abs(t['pl_radeerr1']))
RadiusLSigma = np.array(abs(t['pl_radeerr2']))



StellarMass = np.array(t['st_mass'])
StellarMassUSigma = np.array(t['st_masserr1'])
StellarMassLSigma = np.array(t['st_masserr2'])

# Metallicity = np.array(t['st_met'])
# MetallicitySigma = np.array(t['st_meterr1'])

Insolation = np.array(t['pl_insol'])
InsolationUSigma = np.array(t['pl_insolerr1'])
InsolationLSigma = np.array(t['pl_insolerr2'])


Max, Min = 1, 0
Max, Min = np.nan, np.nan


RadiusDict = {'Data': Radius, 'LSigma': RadiusLSigma,  "USigma":RadiusUSigma, 'Max':Max, 'Min':Min, 'Label':'Radius ($R_{\oplus}$)', 'Char':'r'}
MassDict = {'Data': Mass, 'LSigma': MassLSigma, "USigma":MassUSigma,  'Max':Max, 'Min':Min, 'Label':'Mass ($M_{\oplus}$)', 'Char':'m'}

# PeriodDict = {'Data': Period, 'LSigma': PeriodSigma, "USigma":PeriodSigma, 'Max':Max, 'Min':Min, 'Label':'Period (d)', 'Char':'p'}
StellarMassDict = {'Data': StellarMass, 'LSigma': StellarMassLSigma, "USigma":StellarMassUSigma, 'Max':Max, 'Min':Min, 'Label':'Stellar Mass (M$_{\odot}$)', 'Char':'stm'}
#MetallicityDict = {'Data': 10**Metallicity, 'LSigma': np.repeat(np.nan, len(Metallicity)), "USigma":np.repeat(np.nan, len(Metallicity)), 'Max':np.nan, 'Min':np.nan, 'Label':'Metallicity [Fe/H]', 'Char':'feh'}
# MetallicityDict = {'Data': 10**Metallicity, 'LSigma': np.repeat(np.nan, len(Metallicity)), "USigma":np.repeat(np.nan, len(Metallicity)), 'Max':1, 'Min':-0.45, 'Label':'Metallicity [Fe/H]', 'Char':'feh'}
# MetallicityDict = {'Data': 10**(-Metallicity), 'LSigma': np.repeat(np.nan, len(Metallicity)), "USigma":np.repeat(np.nan, len(Metallicity)), 'Max':np.nan, 'Min':np.nan, 'Label':'Metallicity [Fe/H]', 'Char':'feh'}

InsolationDict = {'Data': Insolation, 'LSigma': InsolationLSigma, "USigma":InsolationUSigma, 'Max':Max, 'Min':Min,  'Label':'Pl Insol ($S_{\oplus}$)', 'Char':'insol'}

from mrexo.mle_utils_nd import InputData, MLE_fit
from mrexo.fit_nd import fit_relation
from mrexo.plotting_nd import Plot2DJointDistribution, Plot2DWeights, Plot1DInputDataHistogram
import matplotlib.pyplot as plt

InputDictionaries = [RadiusDict, MassDict, InsolationDict]
DataDict = InputData(InputDictionaries)

ndim = len(InputDictionaries)
RunName = 'AllPlanet_RpLt20_MRS_test'
save_path = os.path.join(pwd, 'TestRuns',  RunName)

verbose = 2
select_deg = 'aic'
abs_tol = 1e-8
degree_max = 50
cores = 2
NumCandidates = 20
deg_per_dim = [25,25,25]

UseSparseMatrix = True
print("UseSparseMatrix = ",UseSparseMatrix)
s = datetime.datetime.now()
C_pdf = calc_C_matrix(DataDict, deg_per_dim, abs_tol=1e-8, save_path='', verbose=2, UseSparseMatrix=UseSparseMatrix)
w = optimizer(C_pdf, deg_per_dim, verbose=2, save_path='', MaxIter=500, rtol=1e-3, UseSparseMatrix=UseSparseMatrix)
e = datetime.datetime.now()

print(e-s)

print("Using Sparse Matrix = "+str(UseSparseMatrix))
print("C_pdf shape =", np.shape(C_pdf), ':: NumElements x 1e6 = ', np.prod(np.shape(C_pdf))/1e6)
if UseSparseMatrix:
	print("Size for C_pdf in MB = {:.3f}, LogLikelihood = {:.1f}, 95% weight = {}".format(C_pdf.data.size/(1024**2), w[1], np.percentile(w[0], q=95)))
	# print("{:.2f}% of elements are < 1e-10".format(100*np.size(C_pdf.toarray()[C_pdf.toarray() < 1e-10])/np.size(C_pdf.toarray())))
else:
	print("Size for C_pdf in MB = {:.3f}, LogLikelihood = {:.1f}, 95% weight = {}".format(C_pdf.nbytes/(1024**2), w[1], np.percentile(w[0], q=95)))
	# print("{:.2f}% of elements are < 1e-10".format(100*np.size(C_pdf[C_pdf < 1e-10])/np.size(C_pdf)))

"""
ndim = DataDict['ndim']
n = DataDict['DataLength']

degree_candidates = GiveDegreeCandidates(degree_max=degree_max, n=n, ndim=ndim, ncandidates=20)

message = 'Using AIC method to estimate the number of degrees of freedom for the weights. Max candidate = {}\n'.format(degree_candidates.max())
_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)


C_pdf = calc_C_matrix(DataDict, deg_per_dim, abs_tol=1e-8, save_path='', verbose=2, UseSparseMatrix=True)
"""

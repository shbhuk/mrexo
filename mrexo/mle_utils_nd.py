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

import datetime,os
from multiprocessing import current_process
from functools import lru_cache


from .utils_nd import _logging
from .Optimizers import optimizer


########################################
##### Main function: MLE_fit() #########
########################################

# Ndim - 20201130
def InputData(ListofDictionaries):
	"""
	Compile a dictionary of the data and metadata given a list of dictionaries.
	
	Parameters
	----------
	ListofDictionaries : list[dict]
		A list of dictionaries of length 'd', where each dictionary corresponds to a dimension of the data. For example, ``ListofDictionaries = [RadiusDict, MassDict,...]``.
	
	
	The input list should contain dictionaries, each of which contains the following fields:
	
	- `Data`: Data (observables) that are modelled assuming an asymmetric normal distribution.  Length 'L'. The normal distribution consists of two half normals, unnormalized, where the observed measurement sits at the 50% percentile
	- `LSigma`: Sigma value for the lower normal distribution. Same scale as Data (log/linear). Length 'L'
	- `USigma`: Sigma value for the upper normal distribution. Same scale as Data (log/linear). Length 'L'
	- `Max`: log10 of the upper bound for the dimension to be fit. Can leave as np.nan, in which case  = np.log10(1.1*np.max(ndim_data[d]))
	- `Min`: log10 of the lower bound for the dimension to be fit. Can leave as np.nan, in which case  = np.log10(0.9*np.min(ndim_data[d]))
	- `Label`: Axes label (string) to be used for this dimension. E.g. 'Radius ($R_{\oplus}$)' or 'Pl Insol ($S_{\oplus}$)'
	- `Char`: Symbol (character) to be used for this dimension. E.g. 'r' or 's'
	
	
	Returns
	-------
	DataDict : dict
		A dictionary containing all of the data and metadata.
	
	
	The output dictionary ``DataDict`` contains the following fields:
	
	- `ndim_data`: A 2-d array of size (d = number of dimensions, L = number of data points) containing the data.
	- `ndim_sigma`: A 2-d array of size (d, L) containing the uncertainties of each data point (assuming symmetric error bars, taken as the average of the upper and lower uncertainties).
	- `ndim_LSigma`: A 2-d array of size (d, L) containing the lower uncertainties of each data point.
	- `ndim_USigma`: A 2-d array of size (d, L) containing the upper uncertainties of each data point.
	- `ndim_bounds`: A 2-d array of size (d, L) of size (number of dimensions, 2) containing the bounds of the data in each dimension.
	- `ndim_char`: A list of character strings representing the variable in each dimension.
	- `ndim_label`: A list of character strings for labeling the variable in each dimension.
	- `ndim`: The number of dimensions.
	- `DataLength`: The number of data points.
	- `DataSequence`: A 2-d array of size (d, 50) with a uniform sequence for each dimension between the lower and upper bounds. Note: This is uniform in log10-space since the bounds are in log10 space.

	"""
	
	ndim = len(ListofDictionaries)
	ndim_data = np.zeros((ndim, len(ListofDictionaries[0]['Data'])))
	ndim_LSigma  = np.zeros((ndim, len(ListofDictionaries[0]['LSigma'])))
	ndim_USigma  = np.zeros((ndim, len(ListofDictionaries[0]['USigma'])))
	ndim_sigma = np.zeros((ndim, len(ListofDictionaries[0]['USigma'])))
	ndim_bounds = np.zeros((ndim, 2))
	ndim_char = []
	ndim_label = []
	
	for d in range(len(ListofDictionaries)):
		assert len(ListofDictionaries[d]['Data']) == np.shape(ndim_data)[1], "Data entered for dimension {} does not match length for dimension 0".format(d)
		assert len(ListofDictionaries[d]['LSigma']) == np.shape(ndim_LSigma)[1], "Length of Sigma Lower entered for dimension {} does not match length for dimension 0".format(d)
		assert len(ListofDictionaries[d]['USigma']) == np.shape(ndim_USigma)[1], "Length of Sigma Upper entered for dimension {} does not match length for dimension 0".format(d)
		assert len(ListofDictionaries[d]['USigma']) == len(ListofDictionaries[d]['LSigma']), "Length of Sigma Upper entered for dimension {} does not match length for Sigma Lower".format(d)
		assert len(ListofDictionaries[d]['Data']) == len(ListofDictionaries[d]['LSigma']), 'Data and Sigma for dimension {} are not of same length'.format(d)
		
		ndim_data[d] = ListofDictionaries[d]['Data']
		ndim_LSigma[d] = ListofDictionaries[d]['LSigma']
		ndim_USigma[d] = ListofDictionaries[d]['USigma']

		# Anything greater than 50 sigma significance  will use only the Beta function with the sigma values set to NaN
		ndim_LSigma[d][(np.abs(ndim_data[d]/ndim_LSigma[d]) > 50)] = np.nan
		ndim_USigma[d][(np.abs(ndim_data[d]/ndim_USigma[d]) > 50)] = np.nan

		ndim_sigma[d] = np.average([np.abs(ndim_LSigma[d]), np.abs(ndim_USigma[d])], axis=0)
		ndim_char.append(ListofDictionaries[d]['Char'])
		ndim_label.append(ListofDictionaries[d]['Label'])
		
		if not np.isfinite(ListofDictionaries[d]['Min']):
			Min = np.log10(0.9*np.min(np.abs(ndim_data[d])))
		else:
			Min = ListofDictionaries[d]['Min']
			
		if not np.isfinite(ListofDictionaries[d]['Max']):
			Max = np.log10(1.1*np.max(ndim_data[d]))
		else:
			Max = ListofDictionaries[d]['Max']
			 
		ndim_bounds[d] = [Min, Max]
		
	DataDict = {"ndim_data":ndim_data, 
						"ndim_sigma":ndim_sigma,
						"ndim_LSigma":ndim_LSigma,
						"ndim_USigma":ndim_USigma,
						"ndim_bounds":ndim_bounds,
						"ndim_char":ndim_char,
						"ndim_label":ndim_label,
						"ndim":ndim,
						"DataLength":np.shape(ndim_data)[1]}
						
	DataSeq = np.zeros((ndim, 50))
	for dim in range(ndim):
		DataSeq[dim] = np.linspace(DataDict['ndim_bounds'][dim][0], DataDict['ndim_bounds'][dim][1], DataSeq.shape[1]) 
	
	DataDict['DataSequence'] = DataSeq

	return DataDict


def MLE_fit(DataDict, deg_per_dim, 
			abs_tol=1e-8, 
			OutputWeightsOnly=False, CalculateJointDist = False, 
			save_path=None, verbose=2):
	'''
	Perform maximum likelihood estimation to find the weights for the beta density basis functions.
	Also, use those weights to calculate the conditional density distributions.
	Ning et al. 2018 Sec 2.2, Eq 9.

	'''

	starttime = datetime.datetime.now()
	if save_path is None:
		save_path = os.path.dirname(__file__)

	message = '=========Started MLE run at {}\n'.format(starttime)
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

	ndim = DataDict["ndim"]
	DataLength = DataDict["DataLength"]
	
	
	########################################################################
	# Integration to find C matrix (input for log likelihood maximization.)
	########################################################################

	# C_pdf is of shape n x deg_product where deg_product = Product of (deg_i - 2) for i in ndim
	C_pdf = calc_C_matrix(DataDict=DataDict, 
		deg_per_dim=np.array(deg_per_dim), 
		abs_tol=abs_tol, 
		save_path=save_path, 
		verbose=verbose, 
		SaveCMatrix=False)

	message = 'Finished Integration at {}. \nCalculated the PDFs for Integrated beta and normal density.\n'.format(datetime.datetime.now())
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

	###########################################################
	# Run optimization to find the weights
	###########################################################

	# Weights are a 1D vector of length deg_product where deg_product = Product of (deg_i - 2) for i in ndim
	# They can be reshaped into an `ndim` dimensional array 
	unpadded_weight, n_log_lik = optimizer(C_pdf=C_pdf, deg_per_dim=np.array(deg_per_dim),
		verbose=verbose, save_path=save_path)
	
	# unpadded_weight, n_log_lik = SLSQP_optimizer(C_pdf=C_pdf, deg=deg_per_dim[0], 
		# verbose=verbose, save_path=save_path)
	# print("AAAAAAAAAAAAAAAA   {}".format(n_log_lik))
	# rand = np.random.randn()
	# np.savetxt(os.path.join(save_path, 'loglikelihoodtest{:.3f}.txt'.format(rand)), [n_log_lik])
	# np.savetxt(os.path.join(save_path, 'Cpdf{:.3f}.txt'.format(rand)), C_pdf)
	# np.savetxt(os.path.join(save_path, 'IntermediateWeight{:.3f}.txt'.format(rand)), w_hat)


	# Pad the weight array with zeros for the
	w_sq = np.reshape(unpadded_weight, np.array(deg_per_dim)-2)
	w_sq_padded = np.zeros(deg_per_dim)
	w_sq_padded[[slice(1,-1) for i in range(ndim)]] = w_sq
	w_hat = w_sq_padded.flatten()

	if OutputWeightsOnly == True:
		return unpadded_weight, n_log_lik

	else:
		# Calculate AIC and BIC
		deg_product = np.product(deg_per_dim)
		
		NonZero = len(np.nonzero(w_hat)[0])
		Threshold = 1e-8
		#aic = -n_log_lik*2 + 2*(len(w_hat[w_hat>Threshold])/DataLength)
		# aic = -n_log_lik*2 + 2*(NonZero/DataLength)

		# Effective number of weights based on Kish's effective sample size
		EffectiveDOF = (np.sum(w_hat)**2)/np.sum(w_hat**2)
		aic = -n_log_lik*2 + 2*(EffectiveDOF)


		# fi = rank_FI_matrix(C_pdf, unpadded_weight)
		aic_fi = -n_log_lik*2 + 2*(rank_FI_matrix(C_pdf, unpadded_weight))
		# bic = -n_log_lik*2 + np.log(n)*(deg**2 - 1)
		
		DataSeq = DataDict['DataSequence'] 
		
		outputs = {"UnpaddedWeights":unpadded_weight, "Weights":w_hat,
				"loglike":n_log_lik,
				"deg_per_dim":deg_per_dim,
				"DataSequence":DataSeq, 
				"C_pdf":C_pdf,
				"EffectiveDOF": EffectiveDOF,
				"aic":aic, "aic_fi":aic_fi}
		
		if CalculateJointDist:
			JointDist, indv_pdf_per_dim = CalculateJointDistribution(DataDict=DataDict, 
				weights=w_hat, 
				deg_per_dim=deg_per_dim, 
				save_path=save_path, verbose=verbose, abs_tol=abs_tol)
			outputs['JointDist'] = JointDist
		
		return outputs

# Ndim - 20201130
def calc_C_matrix(DataDict, deg_per_dim,
		abs_tol, save_path, verbose, SaveCMatrix=False):
	'''
	Integrate the product of the normal and beta distributions for Y and X and then take the Kronecker product.
	2D matrix with shape = (N x product(degrees-2)). For example in two dimensions this would be (N x (d1-2).(d2-2))

	Refer to Ning et al. 2018 Sec 2.2 Eq 8 and 9.
	'''

	ndim = DataDict['ndim']
	n = DataDict['DataLength']
	# For degree 'd', actually using d-2 since the boundaries are zero padded.
	deg_vec_per_dim = [np.arange(1, deg-1) for deg in deg_per_dim] 
	indv_pdf_per_dim = [np.zeros((n, deg-2)) for deg in deg_per_dim]
	
	# Product of degrees (deg-2 since zero padded)
	deg_product = 1
	for deg in deg_per_dim:
		deg_product *= deg-2
		
	C_pdf = np.zeros((n, deg_product))
	
	message = 'Started Integration at {}\n'.format(datetime.datetime.now())
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

	# Loop across each data point.
	for i in range(0,n):
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
			kron_temp = np.kron(kron_temp, indv_pdf_per_dim[dim][i,:])

		C_pdf[i,:] = kron_temp

	C_pdf = C_pdf.T

	# Log of 0 throws weird errors
	C_pdf[C_pdf <= 0] = 1e-300
	C_pdf[np.where(np.isnan(C_pdf))] = 1e-300

	if SaveCMatrix:
		np.savetxt(os.path.join(save_path, 'C_pdf.txt'), C_pdf)
	return C_pdf


def InvertHalfNormalPDF(x, p, loc=0):
	"""
	Invert the normal PDF to find the sigma that gives '1 - (1-p/2)' as the cumulative probability at 'x'.
	In other words, P(> x) = 1-p

	For example, if you care about 99.7% upper limits, p=0.997. Then we will invert the normal distribution to find the 
	quantile at 1 - (1-0.997/2) = 0.9985, because the upper limits are being approximated as a half-normal,  centered at 0.
	
	INPUTS:
		x: Point at which  P(> x) = 1-p
		p: Probability (decimal)
		loc: Median value of Gaussian Distribution
	OUTPUTS:
		scale: Sigma of Gaussian Distribution
	"""

	p = 1 - (1-p)/2

	scale =  (x-loc)/(erfinv(p*2-1)*np.sqrt(2))
	return scale


def _PDF_Normal(a, loc, scale):
	'''
	Find the PDF for a normal distribution. Identical to scipy.stats.norm.pdf.
	Runs much quicker without the generic function handling.
	'''
	N = (a - loc)/scale
	return (np.e**(-N*N/2))/((2*np.pi)**0.5)/scale # 6x faster
	# return (np.exp(-N*N/2))/(np.sqrt(2*np.pi))/scale

@lru_cache(maxsize=200)
def _GammaFunction(a):
	return scipy.math.factorial(a-1)


def _PDF_Beta(x,a,b):
	'''
	About 50x faster than scipy.stats.beta.pdf
	'''
	if (a>=170) | (b>=170) | (a+b>170):
		f = float((Decimal(_GammaFunction(a+b)) * Decimal(x**(a-1)*(1-x)**(b-1))) / (Decimal(_GammaFunction(a))*Decimal(_GammaFunction(b))))
	else:
		f = (_GammaFunction(a+b) * x**(a-1)*(1-x)**(b-1)) / (_GammaFunction(a)*_GammaFunction(b))

	return f

# Ndim - 20201130
def _PDF_NormalBeta(a, a_obs, a_std, a_max, a_min, shape1, shape2, Log=True):
	'''
	Product of Normal and beta distribution

	Refer to Ning et al. 2018 Sec 2.2, Eq 8.
	'''

	if Log == True: # Integrating in Log Space
		norm_beta = _PDF_Normal(a_obs, loc=10**a, scale=a_std) * _PDF_Beta((a - a_min)/(a_max - a_min), a=shape1, b=shape2)/(a_max - a_min)
	else: # Integrating in Linear Space
		norm_beta = _PDF_Normal(a_obs, loc=a, scale=a_std) * _PDF_Beta((a - a_min)/(a_max - a_min), a=shape1, b=shape2)/(a_max - a_min)
	return norm_beta


# Ndim - 20201130
def IntegrateNormalBeta(data, data_Sigma, deg, degree, a_max, a_min, Log=False, abs_tol=1e-8):
	'''
	Integrate the product of the normal and beta distribution.

	Refer to Ning et al. 2018 Sec 2.2, Eq 8.
	'''
	a_obs = data
	a_std = data_Sigma
	shape1 = degree
	shape2 = deg - degree + 1
	Log = Log

	integration_product = quad(_PDF_NormalBeta, a=a_min, b=a_max,
						  args=(a_obs, a_std, a_max, a_min, shape1, shape2, Log), epsabs=abs_tol, epsrel=1e-8)
	return integration_product[0]


def IntegrateDoubleHalfNormalBeta(data, data_USigma, data_LSigma,
		deg, degree, 
		a_max, a_min, Log=False, abs_tol=1e-8):
	'''
	Integrate the product of the two half normal and beta distribution.

	Refer to Ning et al. 2018 Sec 2.2, Eq 8.
	'''
	a_obs = data
	shape1 = degree
	shape2 = deg - degree + 1
	Log = Log

	if Log: 
		integration_product_L = quad(_PDF_NormalBeta, a=a_min, b=np.log10(a_obs),
							  args=(a_obs, data_LSigma, a_max, a_min, shape1, shape2, Log), epsabs=abs_tol, epsrel=1e-8)
		integration_product_U = quad(_PDF_NormalBeta, a=np.log10(a_obs), b=a_max,
							  args=(a_obs, data_USigma, a_max, a_min, shape1, shape2, Log), epsabs=abs_tol, epsrel=1e-8)
	else:
		integration_product_L = quad(_PDF_NormalBeta, a=a_min, b=a_obs,
							  args=(a_obs, data_LSigma, a_max, a_min, shape1, shape2, Log), epsabs=abs_tol, epsrel=1e-8)
		integration_product_U = quad(_PDF_NormalBeta, a=a_obs, b=a_max,
							  args=(a_obs, data_USigma, a_max, a_min, shape1, shape2, Log), epsabs=abs_tol, epsrel=1e-8)
	return integration_product_L[0] + integration_product_U[0]


# Ndim - 20201130
def _OldComputeConvolvedPDF(a, deg, deg_vec, a_max, a_min, a_LSigma=np.nan, a_USigma=np.nan, abs_tol=1e-8, Log=False):
	'''
	Find the individual probability density function for a variable which is a convolution of a beta function with something else.
	If the data has uncertainty, the joint distribution is modelled using a
	convolution of beta and normal distributions.
	For data with uncertainty, log should be True, because all the computations are done in log space where the observables are considered
	to be in linear space. The individual PDF here is the convolution of a beta and normal function, where the normal distribution captures 
	the measurement uncertainties, the observed quantities, i,e x_obs and x_sigma are in linear space, whereas x the quantity being integrated over is in linear space. 
	Therefore, while calculating the C_pdf, log=True, so that for the PDF of the normal function, everything is in linear space. 
	Conversely, the joint distribution is being computed for the data sequence in logspace , i.e. for x between log(x_min) and log(x_max), and there is no measurement uncertainty. 
	It is taking the weights (already computed considering the measurement uncertainty) and computing the underlying PDF. Therefore, everything is in logspace.
	Refer to Ning et al. 2018 Sec 2.2, Eq 8.
	'''

	a_std = a_LSigma

	if np.isnan(a_std):
		if Log:
			a_std = (np.log10(a) - a_min)/(a_max - a_min)
		else:
			a_std = (a - a_min)/(a_max - a_min)
		a_beta_indv = np.array([_PDF_Beta(a_std, a=d, b=deg - d + 1)/(a_max - a_min) for d in deg_vec])
	else:
		a_beta_indv = np.array([IntegrateNormalBeta(data=a, data_Sigma=a_std, deg=deg, degree=d, a_max=a_max, a_min=a_min, abs_tol=abs_tol, Log=Log) for d in deg_vec])
	return a_beta_indv


def _ComputeConvolvedPDF(a, deg, deg_vec, a_max, a_min, 
	a_LSigma=np.nan, a_USigma=np.nan,
	abs_tol=1e-8, Log=False):
	'''
	Find the individual probability density function for a variable which is a convolution of a beta function with something else.

	If the data has uncertainty, the joint distribution is modelled using a
	convolution of beta and normal distributions. Whereas for data without uncertainty, the convolution basically decomposes to the value of the beta function
	at x = x_obs.

	For data with uncertainty, log should be True, because all the computations are done in log space where the observables are considered
	to be in linear space. The individual PDF here is the convolution of a beta and normal function, where the normal distribution captures 
	the measurement uncertainties, the observed quantities, i,e x_obs and x_sigma are in linear space, whereas x the quantity being integrated over is in log space. 
	Therefore, while calculating the C_pdf, log=True, so that for the PDF of the normal function, everything is in linear space. 

	Conversely, the joint distribution is being computed for the data sequence in logspace , i.e. for x between log(x_min) and log(x_max), and there is no measurement uncertainty. 
	It is taking the weights (already computed considering the measurement uncertainty) and computing the underlying PDF. Therefore, everything is in logspace.

	Refer to Ning et al. 2018 Sec 2.2, Eq 8.

	'''


	if np.isnan(a_USigma) | np.isnan(a_LSigma):
		if Log:
			a_norm = (np.log10(a) - a_min)/(a_max - a_min)
		else:
			a_norm = (a - a_min)/(a_max - a_min)
		PDF = np.array([_PDF_Beta(a_norm, a=d, b=deg - d + 1)/(a_max - a_min) for d in deg_vec])
	else:
		PDF = np.array([IntegrateDoubleHalfNormalBeta(data=a, data_USigma=a_USigma, data_LSigma=a_LSigma, deg=deg, degree=d, a_max=a_max, a_min=a_min, abs_tol=abs_tol, Log=Log) for d in deg_vec])
		# PDF = np.array([IntegrateNormalBeta(data=a, data_Sigma=a_LSigma, deg=deg, degree=d, a_max=a_max, a_min=a_min, abs_tol=abs_tol, Log=Log) for d in deg_vec])


	return PDF


def calculate_conditional_distribution(ConditionString, DataDict, 
		weights, deg_per_dim, 
		JointDist,
		MeasurementDict):
			
	ndim = DataDict['ndim']
	
	Condition = ConditionString.split('|')
	LHSTerms = Condition[0].split(',')
	RHSTerms = Condition[1].split(',')
	deg_vec_per_dim = [np.arange(1, deg+1) for deg in deg_per_dim] 
	
	if len(LHSTerms) == 1:
		ConditionalDist, MeanPDF, VariancePDF = CalculateConditionalDistribution1D_LHS(
			ConditionString, DataDict, 
			weights, deg_per_dim, 
			JointDist,
			MeasurementDict)
	elif len(LHSTerms) == 2:
		ConditionalDist, MeanPDF, VariancePDF = CalculateConditionalDistribution2D_LHS(
			ConditionString, DataDict, 
			weights, deg_per_dim, 
			JointDist,
			MeasurementDict)
			
	return ConditionalDist, MeanPDF, VariancePDF
		

def CalculateConditionalDistribution1D_LHS(ConditionString, DataDict, 
		weights, deg_per_dim, 
		JointDist,
		MeasurementDict):
	'''
	Tested 2021-01-12. Results similar to the old cond_density_quantile() function for one dimension on LHS.
	Based on Eqn 10 from Ning et al. 2018
	INPUTS:
		ConditionString = Example 'x|z or z|y'
		JointDist = An n-dimensional cube with each dimension of same length. Typically 100.
		weights = Padded weights with dimensionality (1 x (d1 x d2 x d3 x .. x dn)) where di are the degrees per dimension
		indv_pdf_per_dim = This is the individual PDF for each point in the sequence . 
			It is a list of 1-D vectors, where the number of elements in the list = ndim.
			The number of elements in each vector is the number of degrees for that dimension
		MeasurementDict: Example:
			MeasurementDict = {}
			MeasurementDict['r'] = [[np.log10(4), np.log10(2)], [np.nan, np.nan]] # Vector of radius measurements, vector of radius measurement uncertainties
			MeasurementDict['p'] = [[np.log10(0.9999), np.log10(1)], [np.nan, np.nan]] # Likewise for period
			
		MeasurementDict assumes log values
	'''
		
	Alphabets = [chr(i) for i in range(105,123)] # Lower case ASCII characters
	ndim = DataDict['ndim']
	
	Condition = ConditionString.split('|')
	LHSTerms = Condition[0].split(',')
	RHSTerms = Condition[1].split(',')
	deg_vec_per_dim = [np.arange(1, deg+1) for deg in deg_per_dim] 

	LHSDimensions = np.array([(np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , l)])[0] for l in LHSTerms])
	RHSDimensions = np.array([(np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , r)])[0] for r in RHSTerms])
	####################################

	# Need to finalize exact structure of input
	RHSMeasurements = []
	RHSUncertainties = []
	_ = [RHSMeasurements.append(MeasurementDict[i][0]) for i in RHSTerms]
	_ = [RHSUncertainties.append(MeasurementDict[i][1]) for i in RHSTerms]
	
	RHSSequence = DataDict['DataSequence'][RHSDimensions]
	LHSSequence = DataDict['DataSequence'][LHSDimensions]
	NSeq = len(RHSSequence[0])
	
	NPoints = len(RHSMeasurements[0])
	ConditionalDist = np.zeros((NPoints, NSeq))
	MeanPDF = np.zeros(NPoints)
	VariancePDF = np.zeros(NPoints)
	
	# Initial values
	ReshapedWeights = np.reshape(weights, deg_per_dim)

	for i in range(NPoints):
		# Indices = [slice(0, None) for _ in range(DataDict['ndim'])]
	
		"""
		# Values at which to perform interpolation
		InterpSlices = np.copy(DataDict['DataSequence'])
		for j in range(len(RHSTerms)):
			# jth RHS dimension, 0 refers to measurement (1 is uncertainty), and taking a slice of the ith measurement input
			# InterpSlices[RHSDimensions[j]] = np.repeat(MeasurementDict[RHSTerms[j]][0][i], NSeq)
			InterpSlices[RHSDimensions[j]] = MeasurementDict[RHSTerms[j]][0][i]
			
		# 20201215 - only works for 1 LHS Dimensions
		SliceofJoint = interpn(tuple(DataDict['DataSequence']), JointDist, np.dstack(InterpSlices))[0]
		"""

		InterpSlices = list(DataDict['DataSequence'])
		for j in range(len(RHSTerms)):
			# jth RHS dimension, 0 refers to measurement (1 is uncertainty), and taking a slice of the ith measurement input
			InterpSlices[RHSDimensions[j]] = MeasurementDict[RHSTerms[j]][0][i]
		
		# Interpolate the joint distribution on to a 1-D grid of points corresponding to the given RHS term to condition on, and sequence of LHS terms that we're querying for.
		# For example to get f(m|r=5), the joint distribution will be interpolated on to (5, m_max), ...,  (5, m_min)
		InterpMesh = np.array(np.meshgrid(*InterpSlices))
		InterpPoints = np.rollaxis(InterpMesh, 0, ndim+1).reshape((NSeq**(len(LHSTerms)), ndim))
		SliceofJoint = interpn(tuple(DataDict['DataSequence']), JointDist, InterpPoints).reshape(tuple(np.repeat(NSeq, len(LHSTerms))))
		
		# Then calculate denominator by taking matrix multiplication
		# Ratio of the two gives the PDF
		# Expectation value of this PDF matches the mean from old (Ning et al. 2018) method
		# Integral(conditionaldist * MassSequence) / Integral(conditionaldist) = Mean(f(m|r)) = Expectation value
		
		temp_denominator = ReshapedWeights
		InputIndices = ''.join(Alphabets[0:DataDict['ndim']])
		 
		for j in range(len(RHSTerms)):
			rdim = RHSDimensions[j]
			indv_pdf = _ComputeConvolvedPDF(a=MeasurementDict[RHSTerms[j]][0][i], 
						a_LSigma=MeasurementDict[RHSTerms[j]][1][i][0], a_USigma=MeasurementDict[RHSTerms[j]][1][i][1],
						deg=deg_per_dim[rdim], deg_vec=deg_vec_per_dim[rdim],
						a_max=DataDict["ndim_bounds"][rdim][1], 
						a_min=DataDict["ndim_bounds"][rdim][0], 
						Log=False) 

			ProductIndices = InputIndices.replace(Alphabets[rdim], '')
			Subscripts = InputIndices+',' +''.join(Alphabets[rdim]) + '->' + ProductIndices

			temp_denominator = TensorMultiplication(temp_denominator, indv_pdf, Subscripts=Subscripts)
			InputIndices = ProductIndices

		# Denominator = np.sum(weights * np.kron(np.ones(deg_per_dim[ldim]), indv_pdf))
		# Denominator = np.sum(np.matmul(ReshapedWeights, indv_pdf))
		
		ConditionalDist[i] = SliceofJoint/np.sum(temp_denominator)
		
		# Find the integral of the Conditional Distribution => Integral of f(x) dx.
		# Should basically -> 1. 
		# 20230102 - Univariate spline is not giving good enough results. Switching to interp1d
		# ConditionPDF = UnivariateSpline(LHSSequence[0], ConditionalDist[i]).integral(
			# DataDict["ndim_bounds"][LHSDimensions[0]][0], DataDict["ndim_bounds"][LHSDimensions[0]][1])
		ConditionPDF = quad(
			func = interp1d(LHSSequence[0], ConditionalDist[i]), 
			a=DataDict["ndim_bounds"][LHSDimensions[0]][0], b=DataDict["ndim_bounds"][LHSDimensions[0]][1]
		)[0]

		# Find the  integral of the product of the Conditional and LHSSequence => E[x]
		# MeanPDF[i] = UnivariateSpline(LHSSequence[0], ConditionalDist[i]*LHSSequence[0]).integral(
			# DataDict["ndim_bounds"][LHSDimensions[0]][0], DataDict["ndim_bounds"][LHSDimensions[0]][1]) / ConditionPDF
		MeanPDF[i] = quad(
			func = interp1d(LHSSequence[0], ConditionalDist[i]*LHSSequence[0]), 
			a=DataDict["ndim_bounds"][LHSDimensions[0]][0], b=DataDict["ndim_bounds"][LHSDimensions[0]][1]
		)[0] / ConditionPDF

		# Variance = E[x^2] - E[x]^2
		# VariancePDF[i] = (UnivariateSpline(LHSSequence[0], ConditionalDist[i]*(LHSSequence[0]**2)).integral(
			# DataDict["ndim_bounds"][LHSDimensions[0]][0], DataDict["ndim_bounds"][LHSDimensions[0]][1])  /  ConditionPDF) - (MeanPDF[i]**2)
		VariancePDF[i] = (quad(
			func = interp1d(LHSSequence[0], ConditionalDist[i]*(LHSSequence[0]**2)), 
			a=DataDict["ndim_bounds"][LHSDimensions[0]][0], b=DataDict["ndim_bounds"][LHSDimensions[0]][1]
		)[0] / ConditionPDF) - (MeanPDF[i]**2)


		# from scipy.integrate import simps
		# print(simps(SliceofJoint, LHSSequence[0]))
		# print(np.sum(temp_denominator))
		# print(ConditionPDF)
		
	return ConditionalDist, MeanPDF, VariancePDF


def CalculateConditionalDistribution2D_LHS(ConditionString, DataDict, 
		weights, deg_per_dim, 
		JointDist,
		MeasurementDict):
	'''
	Verified to work fine for 2 dim in LHS and 1 dim in RHS - 2021-01-12
	INPUTS:
		ConditionString = Example 'x,y|z', or 'm|r,p'
		JointDist = An n-dimensional cube with each dimension of same length. Typically 100.
		weights = Padded weights with dimensionality (1 x (d1 x d2 x d3 x .. x dn)) where di are the degrees per dimension
		indv_pdf_per_dim = This is the individual PDF for each point in the sequence . 
			It is a list of 1-D vectors, where the number of elements in the list = ndim.
			The number of elements in each vector is the number of degrees for that dimension
		MeasurementDict: Example:
			MeasurementDict = {}
			MeasurementDict['r'] = [[np.log10(4), np.log10(2)], [np.nan, np.nan]] # Vector of radius measurements, vector of radius measurement uncertainties
			MeasurementDict['p'] = [[np.log10(0.9999), np.log10(1)], [np.nan, np.nan]] # Likewise for period
			
		MeasurementDict assumes log values
	'''
	
	# Assuming it is x|y,z, i.e. there is only one dimension on the LHS
	
	Alphabets = [chr(i) for i in range(105,123)] # Lower case ASCII characters
	ndim = DataDict['ndim']
	
	# NPoints = len(DataDict['DataSequence'][0])
	Condition = ConditionString.split('|')
	LHSTerms = Condition[0].split(',')
	RHSTerms = Condition[1].split(',')
	deg_vec_per_dim = [np.arange(1, deg+1) for deg in deg_per_dim] 
	
	LHSDimensions = np.array([(np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , l)])[0] for l in LHSTerms])
	RHSDimensions = np.array([(np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , r)])[0] for r in RHSTerms])
	####################################

	# Need to finalize exact structure of input
	RHSMeasurements = []
	RHSUncertainties = []
	_ = [RHSMeasurements.append(MeasurementDict[i][0]) for i in RHSTerms]
	_ = [RHSUncertainties.append(MeasurementDict[i][1]) for i in RHSTerms]
	
	RHSSequence = DataDict['DataSequence'][RHSDimensions]
	LHSSequence = DataDict['DataSequence'][LHSDimensions]
	NSeq = len(RHSSequence[0])
	
	NPoints = len(RHSMeasurements[0])
	ConditionalDist = np.zeros(tuple([NPoints, *np.repeat(NSeq, len(LHSTerms))]))
	MeanPDF = np.zeros((NPoints, len(LHSTerms)))
	VariancePDF = np.zeros((NPoints, len(LHSTerms)))
	
	# Initial values
	ReshapedWeights = np.reshape(weights, deg_per_dim)

	for i in range(NPoints):
		# Indices = [slice(0, None) for _ in range(DataDict['ndim'])]

		"""
		# Values at which to perform interpolation
		InterpSlices = np.copy(DataDict['DataSequence'])
		for j in range(len(RHSTerms)):
			# jth RHS dimension, 0 refers to measurement (1 is uncertainty), and taking a slice of the ith measurement input
			# InterpSlices[RHSDimensions[j]] = np.repeat(MeasurementDict[RHSTerms[j]][0][i], NSeq)
			InterpSlices[RHSDimensions[j]] = MeasurementDict[RHSTerms[j]][0][i]
			
		# 20201215 - only works for 1 LHS Dimensions
		SliceofJoint = interpn(tuple(DataDict['DataSequence']), JointDist, np.dstack(InterpSlices))[0]
		
		#20210310
		We're adding a ~2% error in the ConditionalDistribution 
		when we interpolate the joint distribution
		to obtain the slice along a random measurement value.
		This was tested by checking how close was the integral 
		of the ConditionalDistribution to 1. 
		
		"""

		InterpSlices = list(DataDict['DataSequence'])
		for j in range(len(RHSTerms)):
			# jth RHS dimension, 0 refers to measurement (1 is uncertainty), and taking a slice of the ith measurement input
			InterpSlices[RHSDimensions[j]] = [MeasurementDict[RHSTerms[j]][0][i]]
		
		InterpMesh = np.array(np.meshgrid(*InterpSlices))
		InterpPoints = np.rollaxis(InterpMesh, 0, ndim+1).reshape((NSeq**(len(LHSTerms)), ndim))
		SliceofJoint = interpn(tuple(DataDict['DataSequence']), JointDist.T, InterpPoints).reshape(tuple(np.repeat(NSeq, len(LHSTerms))))
		
		# Hardcoded 20201209
		# Then calculate denominator by taking matrix multiplication
		# Ratio of the two gives the PDF
		# Expectation value of this PDF matches the mean from old (Ning et al. 2018) method
		# Integral(conditionaldist * MassSequence) / Integral(conditionaldist) = Mean(f(m|r)) = Expectation value
		# SliceofJoint = JointDist[RHSIndex, :]
		
		temp_denominator = ReshapedWeights
		InputIndices = ''.join(Alphabets[0:DataDict['ndim']])
		 
		for j in range(len(RHSTerms)):
			rdim = RHSDimensions[j]
			indv_pdf = _ComputeConvolvedPDF(a=MeasurementDict[RHSTerms[j]][0][i], 
						a_LSigma=MeasurementDict[RHSTerms[j]][1][i][0], a_USigma=MeasurementDict[RHSTerms[j]][1][i][1],
						deg=deg_per_dim[rdim], deg_vec=deg_vec_per_dim[rdim],
						a_max=DataDict["ndim_bounds"][rdim][1], 
						a_min=DataDict["ndim_bounds"][rdim][0], 
						Log=False) 

			ProductIndices = InputIndices.replace(Alphabets[rdim], '')
			Subscripts = InputIndices+',' +''.join(Alphabets[rdim]) + '->' + ProductIndices

			temp_denominator = TensorMultiplication(temp_denominator, indv_pdf, Subscripts=Subscripts)
			InputIndices = ProductIndices

		# Denominator = np.sum(weights * np.kron(np.ones(deg_per_dim[ldim]), indv_pdf))
		# Denominator = np.sum(np.matmul(ReshapedWeights, indv_pdf))
		
		ConditionalDist[i] = SliceofJoint/np.sum(temp_denominator)
		
		# Find the integral of the Conditional Distribution => Double Integral of f(x,y) dx dy
		ConditionPDF = RectBivariateSpline(LHSSequence[0], LHSSequence[1], ConditionalDist[i]).integral(
			xa=DataDict["ndim_bounds"][LHSDimensions[0]][0], 
			xb=DataDict["ndim_bounds"][LHSDimensions[0]][1],
			ya=DataDict["ndim_bounds"][LHSDimensions[1]][0], 
			yb=DataDict["ndim_bounds"][LHSDimensions[1]][1])
		
		# Find the  integral of the product of the Conditional and LHSSequence => E[x1, x2] = (E[x1], E[x2])
		MeanPDF[i] = np.array([RectBivariateSpline(LHSSequence[0], LHSSequence[1], ConditionalDist[i]*LHSSequence[j]).integral(
			xa=DataDict["ndim_bounds"][LHSDimensions[0]][0], xb=DataDict["ndim_bounds"][LHSDimensions[0]][1],
			ya=DataDict["ndim_bounds"][LHSDimensions[1]][0], yb=DataDict["ndim_bounds"][LHSDimensions[1]][1]) for j in range(np.shape(LHSSequence)[0])])/ConditionPDF
		
		# Variance = E[x^2] - E[x]^2
		print("Variance not calculated")
		# VariancePDF[i] = (UnivariateSpline(LHSSequence[0], ConditionalDist[i]*(LHSSequence[0]**2)).integral(
			# DataDict["ndim_bounds"][LHSDimensions[0]][0], DataDict["ndim_bounds"][LHSDimensions[0]][1])  /  ConditionPDF) - (MeanPDF[i]**2)
			
	return ConditionalDist, MeanPDF, VariancePDF

def CalculateJointDistribution(DataDict, weights, deg_per_dim, save_path, verbose, abs_tol):
	'''
	# X_points, X_min, X_max, Y_points, Y_min, Y_max, weights, abs_tol):
	Calculcate the joint distribution of Y and X (Y and X) : f(y,x|w,d,d')
	Refer to Ning et al. 2018 Sec 2.1, Eq 7
	
	INPUT:
	weights = Padded weights with dimensionality (1 x (d1 x d2 x d3 x .. x dn)) where di are the degrees per dimension
	
	OUTPUT:
	joint distribution = 
	indv_pdf_per_dim = This is the individual PDF for each point in the sequence . 
		It is a list of 1-D vectors, where the number of elements in the list = ndim.
		The number of elements in each vector is the number of degrees for that dimension
	'''
	ndim = DataDict['ndim']
	
	message = 'Calculating Joint Distribution for {} dimensions at {}\n'.format(ndim, datetime.datetime.now())
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

	
	if ndim==2:
		Joint, indv_pdf_per_dim = CalculateJointDist2D(DataDict, weights, deg_per_dim)
	elif ndim==3:
		Joint, indv_pdf_per_dim = CalculateJointDist3D(DataDict, weights, deg_per_dim)
	elif ndim==4:
		Joint, indv_pdf_per_dim = CalculateJointDist4D(DataDict, weights, deg_per_dim)
		
	message = 'Finished calculating Joint Distribution for {} dimensions at {}\n'.format(ndim, datetime.datetime.now())
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

	return Joint, indv_pdf_per_dim

def CalculateJointDist2D(DataDict, weights, deg_per_dim):
	"""
	Calculate joint distribution for 2D distribution
	"""
	
	# DataSequence is a uniformly distributed (in log space) sequence of equal size for each dimension (typically 100).
	NSeq = len(DataDict['DataSequence'][0])
	deg_vec_per_dim = [np.arange(1, deg+1) for deg in deg_per_dim] 
	ndim = DataDict['ndim']
	Joint = np.zeros([NSeq for i in range(ndim)])
	ReshapedWeights = np.reshape(weights, deg_per_dim)

	indv_pdf_per_dim = [np.zeros((NSeq, deg)) for deg in deg_per_dim]

	# Quick loop to calculate individual PDFs
	for dim in range(ndim):
		for i in range(NSeq):
			indv_pdf_per_dim[dim][i,:] = _ComputeConvolvedPDF(a=DataDict["DataSequence"][dim][i], 
				a_LSigma=np.nan, a_USigma=np.nan,
				deg=deg_per_dim[dim], deg_vec=deg_vec_per_dim[dim],
				a_max=DataDict["ndim_bounds"][dim][1], 
				a_min=DataDict["ndim_bounds"][dim][0], 
				Log=False)
	
	for i in range(NSeq):
		Intermediate1 =  TensorMultiplication(indv_pdf_per_dim[0][i,:], ReshapedWeights)
		for j in range(NSeq):
			Joint[i,j] =  TensorMultiplication(indv_pdf_per_dim[1][j,:], Intermediate1)
			
	return Joint, indv_pdf_per_dim
	

def CalculateJointDist3D(DataDict, weights, deg_per_dim):
	"""
	Calculate joint distribution for 3D distribution
	"""
	# DataSequence is a uniformly distributed (in log space) sequence of equal size for each dimension (typically 100).
	NSeq = len(DataDict['DataSequence'][0])
	deg_vec_per_dim = [np.arange(1, deg+1) for deg in deg_per_dim] 
	ndim = DataDict['ndim']
	Joint = np.zeros([NSeq for i in range(ndim)])
	ReshapedWeights = np.reshape(weights, deg_per_dim)

	indv_pdf_per_dim = [np.zeros((NSeq, deg)) for deg in deg_per_dim]

	# Quick loop to calculate individual PDFs
	for dim in range(ndim):
		for i in range(NSeq):
			indv_pdf_per_dim[dim][i,:] = _ComputeConvolvedPDF(a=DataDict["DataSequence"][dim][i], 
				a_LSigma=np.nan, a_USigma=np.nan,
				deg=deg_per_dim[dim], deg_vec=deg_vec_per_dim[dim],
				a_max=DataDict["ndim_bounds"][dim][1], 
				a_min=DataDict["ndim_bounds"][dim][0], 
				Log=False)
	"""
	for i in range(NSeq):
		Intermediate1 =  TensorMultiplication(indv_pdf_per_dim[0][i,:], ReshapedWeights)
		for j in range(NSeq):
			Intermediate2 =  TensorMultiplication(indv_pdf_per_dim[1][j,:], Intermediate1)
			for k in range(NSeq):
				Joint[i,j,k] = TensorMultiplication(indv_pdf_per_dim[2][k,:], Intermediate2)
	"""
	
	for i in range(NSeq):
		Intermediate1 =  TensorMultiplication(ReshapedWeights, indv_pdf_per_dim[2][i,:])
		for j in range(NSeq):
			Intermediate2 =  TensorMultiplication(Intermediate1, indv_pdf_per_dim[1][j,:])
			for k in range(NSeq):
				Joint[i,j,k] = TensorMultiplication(Intermediate2, indv_pdf_per_dim[0][k,:])
	
	return Joint, indv_pdf_per_dim


def CalculateJointDist4D(DataDict, weights, deg_per_dim):
	"""
	Calculate joint distribution for 4D distribution
	"""
	# DataSequence is a uniformly distributed (in log space) sequence of equal size for each dimension (typically 100).
	NSeq = len(DataDict['DataSequence'][0])
	deg_vec_per_dim = [np.arange(1, deg+1) for deg in deg_per_dim] 
	ndim = DataDict['ndim']
	Joint = np.zeros([NSeq for i in range(ndim)])
	ReshapedWeights = np.reshape(weights, deg_per_dim)

	indv_pdf_per_dim = [np.zeros((NSeq, deg)) for deg in deg_per_dim]

	# Quick loop to calculate individual PDFs
	for dim in range(ndim):
		for i in range(NSeq):
			indv_pdf_per_dim[dim][i,:] = _ComputeConvolvedPDF(a=DataDict["DataSequence"][dim][i], 
				a_LSigma=np.nan, a_USigma=np.nan,
				deg=deg_per_dim[dim], deg_vec=deg_vec_per_dim[dim],
				a_max=DataDict["ndim_bounds"][dim][1], 
				a_min=DataDict["ndim_bounds"][dim][0], 
				Log=False)

	for i in range(NSeq):
		Intermediate1 =  TensorMultiplication(indv_pdf_per_dim[0][i,:], ReshapedWeights)
		for j in range(NSeq):
			Intermediate2 =  TensorMultiplication(indv_pdf_per_dim[1][j,:], Intermediate1)
			for k in range(NSeq):
				Intermediate3 = TensorMultiplication(indv_pdf_per_dim[2][k,:], Intermediate2)
				for l in range(NSeq):
					Joint[i,j,k, l] = TensorMultiplication(indv_pdf_per_dim[3][l,:], Intermediate3)
	"""
	for i in range(NSeq):
		Intermediate1 =  TensorMultiplication(ReshapedWeights, indv_pdf_per_dim[1][i,:])
		for j in range(NSeq):
			Intermediate2 =  TensorMultiplication(Intermediate1, indv_pdf_per_dim[1][j,:])
			for k in range(NSeq):
				Intermediate3 =  TensorMultiplication(Intermediate2, indv_pdf_per_dim[1][k,:])
				for l in range(NSeq):
					Joint[i,j,k, l] = TensorMultiplication(Intermediate3, indv_pdf_per_dim[0][l,:])
	"""		
	return Joint, indv_pdf_per_dim


def _OldCalculateJointDist2D(DataDict, weights, deg_per_dim):
	'''
	Much slower than the previous one
	'''
	# DataSequence is a uniformly distributed (in log space) sequence of equal size for each dimension (typically 100).
	NSeq = len(DataDict['DataSequence'][0])
	deg_vec_per_dim = [np.arange(1, deg+1) for deg in deg_per_dim] 
	ndim = DataDict['ndim']
	Joint = np.zeros([NSeq for i in range(ndim)])
	ReshapedWeights = np.reshape(weights, deg_per_dim)

	indv_pdf_per_dim = [np.zeros((NSeq, deg)) for deg in deg_per_dim]

	for i in range(NSeq):
		indv_pdf_per_dim[0][i,:] = _ComputeConvolvedPDF(a=DataDict["DataSequence"][0][i], 
				a_LSigma=np.nan, a_USigma=np.nan,
			deg=deg_per_dim[0], deg_vec=deg_vec_per_dim[0],
			a_max=DataDict["ndim_bounds"][0][1], 
			a_min=DataDict["ndim_bounds"][0][0], 
			Log=False)
		for j in range(NSeq):
			indv_pdf_per_dim[1][j,:] = _ComputeConvolvedPDF(a=DataDict["DataSequence"][1][j], 
				a_LSigma=np.nan, a_USigma=np.nan,
				deg=deg_per_dim[1], deg_vec=deg_vec_per_dim[0],
				a_max=DataDict["ndim_bounds"][1][1], 
				a_min=DataDict["ndim_bounds"][1][0], 
				Log=False)
			Intermediate = TensorMultiplication(ReshapedWeights, indv_pdf_per_dim[1][j,:])
			Joint[i,j] = TensorMultiplication(Intermediate, indv_pdf_per_dim[0][i,:])
	
	return Joint



def _OldCalculateJointDistribution(DataDict, weights, deg_per_dim, save_path, verbose, abs_tol):
	'''
	'''
	
	message = 'Calculating Joint Distribution at {}'.format(datetime.datetime.now())
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

	# DataSequence is a uniformly distributed (in log space) sequence of equal size for each dimension (typically 100).
	NSeq = len(DataDict['DataSequence'][0])
	deg_vec_per_dim = [np.arange(1, deg+1) for deg in deg_per_dim] 
	ndim = DataDict['ndim']

	joint = np.zeros([NSeq for i in range(ndim)])
	indv_pdf_per_dim = [np.zeros((NSeq, deg)) for deg in deg_per_dim]
	Indices = np.zeros(ndim, dtype=int)
	
	def JointRecursiveMess(dim):
		for i in range(NSeq):
			Indices[dim] =  i
			
			# Here log is false, since the measurements are drawn from DataSequence which is uniformly 
			# distributed in log space (between Max and Min)
			indv_pdf_per_dim[dim][i,:] = _ComputeConvolvedPDF(a=DataDict["DataSequence"][dim][i], 
				a_LSigma=np.nan, a_USigma=np.nan,
				deg=deg_per_dim[dim], deg_vec=deg_vec_per_dim[dim],
				a_max=DataDict["ndim_bounds"][dim][1], 
				a_min=DataDict["ndim_bounds"][dim][0], 
				Log=False)
				
			# print(Indices, dim, np.sum(indv_pdf_per_dim[dim][i,:]))
				
			if dim == ndim-1:
				temporary = np.reshape(weights, deg_per_dim)
				# Perform matrix multiplication to multiply the individual PDFs with the weights and obtain the Joint Distribution
				for dd in range(ndim):
					# 20201226 - Check if this multiplication is correct. Should it be Indices[dd] or something else?x
					# print(Indices, dd, np.sum(indv_pdf_per_dim[dd][Indices[dd],:]), np.sum(temporary), joint[tuple(Indices)])
					temporary = TensorMultiplication(indv_pdf_per_dim[dd][Indices[dd],:], temporary)

				joint[tuple(Indices)] = temporary
				# print(joint[tuple(Indices)])
			else:
				# print("Init next loop for ", Indices, dim+1)
				JointRecursiveMess(dim+1)
				
	JointRecursiveMess(0)

	message = 'Finished calculating Joint Distribution at {}'.format(datetime.datetime.now())
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

	return joint, indv_pdf_per_dim


def __OldCalculateJointDistribution(DataDict, weights, deg_per_dim, save_path, verbose, abs_tol):
	'''

	'''
	
	# 20201226 - Try using a different method
	
	message = 'Calculating Joint Distribution at {}'.format(datetime.datetime.now())
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

	# DataSequence is a uniformly distributed (in log space) sequence of equal size for each dimension (typically 100).
	NSeq = len(DataDict['DataSequence'][0])
	deg_vec_per_dim = [np.arange(1, deg+1) for deg in deg_per_dim] 
	ndim = DataDict['ndim']

	joint = np.zeros([NSeq for i in range(ndim)])
	indv_pdf_per_dim = [np.zeros((NSeq, deg)) for deg in deg_per_dim]

	Indices = np.zeros(ndim, dtype=int)
	for i in range(NSeq):
		for dim in range(ndim):
			Indices[dim] = i
			
			# Here log is false, since the measurements are drawn from DataSequence which is uniformly 
			# distributed in log space (between Max and Min)
			indv_pdf_per_dim[dim][i,:] = _ComputeConvolvedPDF(a=DataDict["DataSequence"][dim][i], 
				a_LSigma=np.nan, a_USigma=np.nan,
				deg=deg_per_dim[dim], deg_vec=deg_vec_per_dim[dim],
				a_max=DataDict["ndim_bounds"][dim][1], 
				a_min=DataDict["ndim_bounds"][dim][0], 
				Log=False)

		temporary = np.reshape(weights, deg_per_dim)
		#Perform matrix multiplication to multiply the individual PDFs with the weights and obtain the Joint Distribution
		for dd in range(ndim):
			temporary = TensorMultiplication(indv_pdf_per_dim[dd][Indices[dd],:], temporary)

		joint[tuple(Indices)] = temporary

	message = 'Finished calculating Joint Distribution at {}'.format(datetime.datetime.now())
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

	return joint, indv_pdf_per_dim



def TensorMultiplication(A, B, Subscripts=None):
	"""
	Calculate the tensor contraction of A and B.
	If Subscripts = None: Calculate along the last dimension of A, and the first dimension of B 
		which must match in size.
		C[i,j,...,l,...,n] = A[i,j,...,k] * B[k,l,...n]
		NOTE: if dimension n is length 1, C is reshaped to reduce that dimension.
	else:
		Provide Subscripts example: 'ijk,klm -> ijlm'
	"""
	
	
	Alphabets = [chr(i) for i in range(105,123)] # Lower case ASCII characters
	NdimA = A.ndim
	NdimB = B.ndim
	
	if Subscripts is None:
		Subscripts = ''.join(Alphabets[0:NdimA])+ ',' +\
			''.join(Alphabets[NdimA-1:NdimA+NdimB-1]) + '->' +\
			 ''.join(Alphabets[0:NdimA-1]+Alphabets[NdimA:NdimA+NdimB-1])
	return np.einsum(Subscripts, A, B)


def rank_FI_matrix(C_pdf, w):
	"""
	Compute the Rank of the Fisher Information Matrix as an estimate of the DoFs for AIC.

	INPUT:
		C_pdf: 2d array with [n, (deg-2)^2]
		w: 2D array with [deg-2, deg-2]
	OUTPUT:
		Rank (Integer)
	"""
	n = np.shape(C_pdf)[1] # number of data points

	score = C_pdf/np.matmul(C_pdf.T, w)
	FI = np.matmul(score, score.T/n)
	Rank = np.linalg.matrix_rank(FI)

	return Rank

def _rank_FI_matrix(C_pdf, w):
	"""
	INPUT:
	C_pdf: 2d array with [n, (deg-2)^2]
	w: 2D array with [deg-2, deg-2]

	"""
	n = np.shape(C_pdf)[1] # number of data points
	deg_min2_sq = np.shape(C_pdf)[0]

	#C_pdf_transpose = transpose(C_pdf)

	F = np.zeros((deg_min2_sq, deg_min2_sq))

	start = datetime.datetime.now()
	for i in range(n):
		F += np.outer(C_pdf[:,i], C_pdf[:,i].T) / ((C_pdf[:,i].T * w)**2)
	end = datetime.datetime.now()
	print(end-start)
   	#F += reshape(kron(C_pdf[:,i], C_pdf_transpose[i,:]), (deg_min2_sq, deg_min2_sq)) / sum(C_pdf_transpose[i,:] .* w)^2
   	#println(i)

	F = F/n
	return F
	

def NumericalIntegrate2D(xarray, yarray, Matrix, xlimits, ylimits):
	"""
	
	
	"""
	
	Integral = RectBivariateSpline(xarray, yarray, Matrix).integral(
		xa=xlimits[0], xb=xlimits[1], ya=ylimits[0], yb=ylimits[1])
	# Integral2 = simps(simps(Matrix, xarray), yarray)
	return Integral


def ObtainScipyPDF(xseq, PDF):
	"""
	Take PDF as a function of xseq and convert it into a scipy class object in rv_continuous

	"""

	Inter = interp1d(xseq, PDF, bounds_error=False)
	class pdf_gen(rv_continuous):
		def _pdf(self, x):
			return Inter(x)

	CustomPDF = pdf_gen("CustomPDF")

	# Re-define the bounds to not be +- inf
	CustomPDF.a = xseq[0]
	CustomPDF.b = xseq[-1]

	# Can then obtain a random sample as follows
	# sample = CustomPDF.rvs(size=500)

	return CustomPDF
	

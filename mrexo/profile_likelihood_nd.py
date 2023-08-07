# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd
from multiprocessing import Pool
from .mle_utils_nd import MLE_fit, calc_C_matrix
from .utils_nd import _logging, GiveDegreeCandidates



def run_profile_likelihood(DataDict, degree_max, cores=1,
	save_path=os.path.dirname(__file__), verbose=2, abs_tol=1e-8):
	"""
	Choose the number of degrees for each dimension using the profile likelihood.
	
	Parameters
	----------
	DataDict : dict
		The dictionary containing the data. See the output of :py:func:`mrexo.mle_utils_nd.InputData`.
	degree_max : int or array[int]
		The maximum number of degrees to use for each dimension (the same value is used for all dimensions if an integer is provided.
	cores : int, default=1
		The number of cores to perform the calculation with. Defaults to a single core (serial implementation).
	save_path : str, default=os.path.dirname(__file__)
		The folder name (including path) to save results in.
	verbose : int, default=2
		Integer specifying verbosity for logging: 0 (will not log in the log file or print statements), 1 (will write log file only), or 2 (will write log file and print statements).
	abs_tol : float, default=1e-8
		The absolute tolerance to be used for the numerical integration for the product of normal and beta distributions.
	
	Returns
	-------
	deg_choose : array[int]
		The number of degrees for each dimension, chosen using the profile likelihood.
	"""

	ndim = DataDict['ndim']
	n = DataDict['DataLength']

	degree_candidates = GiveDegreeCandidates(degree_max, n, ndim)
	ncand_per_dim = tuple(len(degs) for degs in degree_candidates) # NOTE: currently, 'degree_candidates' will always have the same number of candidates per dimension, but this line is written more generally
	#degcand_grids = np.meshgrid(*degree_candidates) # list of 'ndim' arrays, each of size 'ncand_per_dim'

	message = "Running profile likelihood method to estimate the number of degrees of freedom for the weights. Max candidates = {}\n".format(degree_candidates.max())
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

	##### NOTE: this was used in 1 dimension; for higher dimensions, can either use this for the magnitude of the gradient, or do something else
	# For profile likelihood to pick the optimum number of degrees,
	# Define tolerance to 0.01 or 1/n whichever is larger.
	# Do not want arbitrarily small tolerances for large sample sizes, and
	# for small samples 1% might never be met.
	# logliketolerance = max(0.01, 1/n)

	if cores>1:
		message = "Running profile likelihood in parallel"
		print('WARNING: profile likelihood in parallel has not been implemented yet!')
		# TODO: implement profile likelihood in parallel
	else:
		message = "Running profile likelihood in serial"
		_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)
		
		loglike = np.zeros(ncand_per_dim)
		
		for idx, _ in np.ndenumerate(loglike): # 'idx' is a tuple of indices for each dimension
			_, loglike[idx] = MLE_fit(DataDict, deg_per_dim=[degree_candidates[d][idx[d]] for d in range(ndim)], abs_tol=abs_tol, save_path=save_path, OutputWeightsOnly=True, CalculateJointDist=False, verbose=verbose)

	Gradient = np.gradient(loglike, *degree_candidates)
	TotalDerivative = (Gradient[0]**2 + Gradient[1]**2) # QUESTION: is this really the "total derivative"? It looks more like the magnitude of the gradient.
	deg_choose = ... # TODO: figure out how to choose the degrees in each dimension

	# NOTE: imshow only works in two dimensions (ndim=2)
	'''
	plt.imshow(loglike, origin='lower', interpolation='bicubic')
	plt.colorbar()
	plt.title(degree_max)
	plt.show(block=False)
	'''

	df = pd.DataFrame({"degree_candidates":degree_candidates, "LogLikelihood":loglike, "FractionalChange":FractionalChange})
	df.to_csv(os.path.join(save_path, 'likelihood_per_degree.csv'), index=False)

	message='Finished Profile Likelihood. Picked {} degrees.'.format(deg_choose)
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

	return deg_choose


# Completely rewrite the function below, if needed when implementing the parallel version of the profile likelihood calculation:
'''
def _profilelikelihood_parallelize(pl_input):

	Y, X, Y_sigma, X_sigma, Y_bounds, X_bounds, Y_char, X_char, deg, abs_tol, save_path, verbose = pl_input

	message = "\nRunning profile likelihod for deg = "+str(deg)
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)


	_, loglike = MLE_fit(Y=Y, X=X, Y_sigma=Y_sigma, X_sigma=X_sigma,
			Y_bounds=Y_bounds, X_bounds=X_bounds, Y_char=Y_char, X_char=X_char,
			deg=deg, abs_tol=abs_tol, save_path=save_path, output_weights_only=True, verbose=verbose)
	return loglike
'''



""" OLD VERSIONS:
def run_profile_likelihood(Y, X, Y_sigma, X_sigma, Y_bounds, X_bounds,
						X_char='x', Y_char='y',
						degree_max=None, degree_candidates=None,
						cores=1, save_path=os.path.dirname(__file__), verbose=2, abs_tol=1e-8):
	#'''
	\nINPUTS:
		Y: Numpy array of Y measurements. In LINEAR SCALE.
		X: Numpy array of X measurements. In LINEAR SCALE.
		Y_sigma: Numpy array of Y uncertainties. Assumes symmetrical uncertainty. In LINEAR SCALE.
		X_sigma: Numpy array of X uncertainties. Assumes symmetrical uncertainty. In LINEAR SCALE.
		Y_bounds: Bounds for the Y. Log10
		X_bounds: Bounds for the X. Log10
		X_char: String alphabet (character) to depict X quantity.
			Eg 'm' for Mass, 'r' for Radius
		Y_char: String alphabet (character) to depict Y quantity
			Eg 'm' for Mass, 'r' for Radius
		degree_max: Maximum degree used for cross-validation/AIC/BIC. Type: Integer.
					Default=None. If None, uses: n/np.log10(n), where n is the number of data points.
		k_fold: If using cross validation method, use k_fold (integer) number of folds. Default=None.
				If None, uses:
				  - 10 folds for n > 60, where n is the length of the Y and X arrays.
				  - Uses 5 folds otherwise.
		degree_candidates: Integer vector containing degrees to run cross validation check for. Default is None.
					If None, defaults to 12 values between 5 and degree_max.
		cores: Number of cores for parallel processing. This is used in the
				bootstrap and the cross validation. Default=1.
				To use all the cores in the CPU, cores=cpu_count() (from multiprocessing import cpu_count)
		abs_tol : Absolute tolerance to be used for the numerical integration for product of normal and beta distribution.
				Default : 1e-8
		cores: this program uses parallel computing for bootstrap. Default=1
		save_path: Location of folder within results for auxiliary output files
		verbose: Integer specifying verbosity for logging.
		If 0: Will not log in the log file or print statements.
		If 1: Will write log file only.
		If 2: Will write log file and print statements.

	OUTPUTS:

		deg_choose - The optimum degree chosen by cross validation and MLE
	#'''

	n = len(Y)


	# For small samples (<250), use a uniformly spaced grid for degrees, otherwise,
	# uniformly spaced in powers of n.

	if degree_candidates == None:
		if n < 250:
			if degree_max == None:
				degree_max = int(n/np.log(n)) + 2
			else:
				degree_max = int(degree_max)
			degree_candidates = np.unique(np.linspace(5, degree_max, 10, dtype = int))
		else:
			if degree_max == None:
				degree_candidates = (np.floor(n**np.linspace(0.3, 0.76, 10))).astype(int)
			else:
				degree_max = int(degree_max)
				degree_candidates = (np.floor(n**np.linspace(0.3, np.log(degree_max)/np.log(n), 10))).astype(int)
	else:
		degree_candidates = np.sort(degree_candidates)



	message = 'Running profile likelihood method to estimate the number of degrees of freedom for the weights. Max candidate = {}\n'.format(degree_candidates.max())
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

	# For profile likelihood to pick the optimum number of degrees,
	# Define tolerance to 0.01 or 1/n whichever is larger.
	# Do not want arbitrarily small tolerances for large sample sizes, and
	# for small samples 1% might never be met.
	logliketolerance = max(0.01, 1/n)

	if cores>1:
		message = "Running profile likelihood in parallel"
		_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

		pl_input = ((Y, X, Y_sigma, X_sigma, Y_bounds, X_bounds, Y_char, X_char,
					d, abs_tol, save_path, verbose) for d in degree_candidates)

		# Run in parallel
		pool = Pool(processes = cores)
		pl_result = list(pool.imap(_profilelikelihood_parallelize, pl_input))
		loglike = np.array(pl_result)

		FractionalChange = np.diff(loglike)/np.abs(loglike[1:])
		Mask = FractionalChange[:] <= logliketolerance

		if len(degree_candidates[1:][Mask]) == 0:
			deg_choose = degree_candidates.max()

			message = "Threshold not reached for maximum degree="+str(degree_candidates.max())
			_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)
		else:
			deg_choose = np.min(degree_candidates[1:][Mask])

		FractionalChange = [1, *FractionalChange]
		# df = pd.DataFrame({"degree_candidates":degree_candidates[1:], "LogLikelihood":loglike[1:], "FractionalChange":FractionalChange})
		# df.to_csv(os.path.join(save_path, 'likelihood_per_degree.csv'), index=False)

	else:
		message = "Running profile likelihood in serial"
		_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

		loglike = np.zeros(len(degree_candidates))
		FractionalChange= np.ones(len(degree_candidates))

		t = 0

		while FractionalChange[t] > logliketolerance:

			# Calculate the optimum weights using MLE for a given input test_degree
			_, loglike[t] = MLE_fit(Y=Y, X=X, Y_sigma=Y_sigma, X_sigma=X_sigma,
					Y_bounds=Y_bounds, X_bounds=X_bounds, Y_char=Y_char, X_char=X_char,
					deg=degree_candidates[t], abs_tol=abs_tol, save_path=save_path, output_weights_only=True, verbose=verbose)
			if t > 0:
				FractionalChange[t] = (loglike[t] - loglike[t-1])/np.abs(loglike[t-1])
				print(t, degree_candidates[t], loglike[t], FractionalChange[t])
				deg_choose = degree_candidates[t]


			if t == len(degree_candidates)-1:
				message = "Threshold not reached for maximum degree="+str(degree_candidates.max())
				_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

				break
			else:
				t+=1


	df = pd.DataFrame({"degree_candidates":degree_candidates, "LogLikelihood":loglike, "FractionalChange":FractionalChange})
	df.to_csv(os.path.join(save_path, 'likelihood_per_degree.csv'), index=False)

	message='Finished Profile Likelihood. Picked {} degrees.'.format(deg_choose)
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)




	return deg_choose


def _profilelikelihood_parallelize(pl_input):

	Y, X, Y_sigma, X_sigma, Y_bounds, X_bounds, Y_char, X_char, deg, abs_tol, save_path, verbose = pl_input

	message = "\nRunning profile likelihod for deg = "+str(deg)
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)


	_, loglike = MLE_fit(Y=Y, X=X, Y_sigma=Y_sigma, X_sigma=X_sigma,
			Y_bounds=Y_bounds, X_bounds=X_bounds, Y_char=Y_char, X_char=X_char,
			deg=deg, abs_tol=abs_tol, save_path=save_path, output_weights_only=True, verbose=verbose)
	return loglike
"""

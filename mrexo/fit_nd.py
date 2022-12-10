#%cd "C:/Users/shbhu/Documents/Git/Py_Y_X_working/PyCode"
import numpy as np
from multiprocessing import Pool,cpu_count
import os
from astropy.table import Table
import datetime

from .mle_utils_nd import MLE_fit
from .cross_validate_nd import run_cross_validation
from .profile_likelihood import run_profile_likelihood
from .utils_nd import _save_dictionary, _logging
from .aic_nd import run_aic



def fit_relation(DataDict, SigmaLimit=1e-3, 
		save_path=None, select_deg=None, degree_max=None, SymmetricDegreePerDimension=True, 
		k_fold=None, num_boot=100, cores=1, abs_tol=1e-8, verbose=2):
	"""
	Fit an n-dimensional relationship using a non parametric model with beta densities.

	Parameters
	----------
	DataDict : dict
		The dictionary containing the data. See the output of :py:func:`mrexo.mle_utils_nd.InputData`.
	SigmaLimit : int, default=1e-3
		The lower limit on the sigma values for all dimensions. Sigma values lower than this limit will be changed to None. This is required because the standard normal distribution blows up if the sigma values are too small (~1e-4). Then the distribution is no longer a convolution of normal and beta distributions, but is just a beta distribution.
	save_path : str, optional
		The folder name (including path) to save results in. For example, ``save_path = '~/mrexo_working/trial_result'`` will create the 'trial_result' folder in 'mrexo_working' to contain the results.
	select_deg : {'cv', 'aic', 'bic'} or int, optional
		The number of degrees (or method of determining the number of degrees) for the beta densities. If "cv", will use cross validation to find the optimal number of  degrees. If "aic", will use AIC minimization. If "bic", will use BIC minimization. If an integer, will use that number and skip the optimization process for the number of degrees. NOTE: Use AIC or BIC optimization only for large (>200) sample sizes.
	degree_max : int, optional
		The maximum degree checked during degree selection. By default, uses ``n/np.log10(n)``, where ``n`` is the number of data points.
	SymmetricDegreePerDimension: boolean, default=True
		If True, while optimizing the number of degrees, it assumes the same number of degrees in each dimension (i.e. symmetric).
		In the symmetric case, it runs through ``NumCandidates`` iterations, typically 20. So the degree candidates are [d1, d1], [d2, d2], etc..
		If False, while optimizing the number of degrees it can have ``NumCandidates ^ NumDimensions`` iterations. Therefore with 20 degree candidates in 2 dimensions, there will be 400 iterations to go through!
	k_fold : int, optional
		The number of folds, if using k-fold validation. Only used if ``select_deg='cv'``. By default, uses 10 folds for n > 60, and 5 folds otherwise.
	num_boot : int, default=100
		The number of bootstraps to perform (must be greater than 1).
	cores : int, default=1
		The number of cores to use for parallel processing. This is used in the
		   bootstrap and the cross validation. To use all the cores in the CPU,
		   set ``cores=cpu_count()`` (requires '#from multiprocessing import cpu_count').
	abs_tol : float, default=1e-8
		The absolute tolerance to be used for the numerical integration for the product of normal and beta distributions.
	verbose : int, default=2
		Integer specifying verbosity for logging: 0 (will not log in the log file or print statements), 1 (will write log file only), or 2 (will write log file and print statements).

	Returns
	-------
	initialfit_result : dict
		Output dictionary from initial fitting without bootstrap using Maximum Likelihood Estimation. See the output of :py:func:`mrexo.mle_utils_nd.MLE_fit`.
	bootstrap_results : dict
		TBD. Only returned if ``num_boot`` > 0.
	
	"""

	starttime = datetime.datetime.now()


	print('Started for {} degrees at {}, using {} core/s'.format(select_deg, starttime, cores))

	# Create subdirectories for results
	input_location = os.path.join(save_path, 'input')
	output_location = os.path.join(save_path, 'output')
	aux_output_location = os.path.join(output_location, 'other_data_products')

	if not os.path.exists(save_path):
		os.mkdir(save_path)
	if not os.path.exists(output_location):
		os.mkdir(output_location)
	if not os.path.exists(aux_output_location):
		os.mkdir(aux_output_location)
	if not os.path.exists(input_location):
		os.mkdir(input_location)

	# LabelDictionary = {'X_label':X_label, 'Y_label':Y_label, 'X_char': X_char, 'Y_char':Y_char}
	Labels = DataDict['ndim_label']
	Char = DataDict['ndim_char']
	np.savetxt(os.path.join(aux_output_location, 'NDimLabels.txt'), Labels, fmt="%s", comments="#Dimension Labels")
	np.savetxt(os.path.join(aux_output_location, 'NDimChar.txt'), Char, fmt="%s", comments="#Dimension Character")


	message = """
	___  _________ _______   _______
	|  \/  || ___ \  ___\ \ / /  _  |
	| .  . || |_/ / |__  \ V /| | | |
	| |\/| ||	/|  __| /   \| | | |
	| |  | || |\ \| |___/ /^\ \ \_/ /
	\_|  |_/\_| \_\____/\/   \/\___/
	"""

	_ = _logging(message=message, filepath=aux_output_location, verbose=verbose, append=True)

	message = 'Started for {} degrees at {}, using {} core/s'.format(select_deg, starttime, cores)
	_ = _logging(message=message, filepath=aux_output_location, verbose=verbose, append=True)

	print("Currently not including sigma limit")
	# Y_sigma[(Y_sigma!=np.nan) & (Y_sigma[Y_sigma!=np.nan] < YSigmaLimit)] = np.nan
	# X_sigma[(X_sigma!=np.nan) & (X_sigma[X_sigma!=np.nan] < XSigmaLimit)] = np.nan

	np.save(os.path.join(input_location, 'DataDict.npy'), DataDict)
	
	###########################################################
	## Step 1: Select number of degrees based on cross validation (CV), AIC or BIC methods.
	print(select_deg)
	if select_deg == 'cv':

		deg_per_dim = run_cross_validation(DataDict, degree_max, k_fold=10, NumCandidates=20, 
			SymmetricDegreePerDimension=SymmetricDegreePerDimension,
			cores=cores, save_path=aux_output_location, verbose=verbose, abs_tol=abs_tol)

	elif select_deg == 'profile':
		print("Profile Likelihood is not implemented yet")
	elif select_deg == 'aic' :

		deg_per_dim = run_aic(DataDict, degree_max, NumCandidates=20, 
			SymmetricDegreePerDimension=SymmetricDegreePerDimension,
			cores=cores, save_path=aux_output_location, verbose=verbose, abs_tol=abs_tol)

	else:
		# Use user defined value
		deg_per_dim = select_deg

	###########################################################
	## Step 2: Estimate the full model without bootstrap

	print('Running full dataset MLE before bootstrap')


	message = 'Running full dataset MLE before bootstrap\n'
	_ = _logging(message=message, filepath=aux_output_location, verbose=verbose, append=True)

	initialfit_result = MLE_fit(DataDict,  deg_per_dim=deg_per_dim,
	save_path=save_path, verbose=verbose, abs_tol=abs_tol,
	OutputWeightsOnly=False, CalculateJointDist=True)


	message = 'Finished full dataset MLE run at {}\n'.format(datetime.datetime.now())
	_ = _logging(message=message, filepath=aux_output_location, verbose=verbose, append=True)

	_save_dictionary(dictionary=initialfit_result, output_location=output_location, bootstrap=False)

	###########################################################
	## Step 3: Run Bootstrap
	if num_boot == 0:
		message='Bootstrap not run since num_boot = 0'
		_ = _logging(message=message, filepath=aux_output_location, verbose=verbose, append=True)
		return initialfit_result, _
	else:
		print("Bootstrapping hasn't been coded up")
		message = """
	 _______ _	_ ______   ______ _   _ _____
	 |__   __| |  | |  ____| |  ____| \ | |  __ |
		| |  | |__| | |__	| |__  |  \| | |  | |
		| |  |  __  |  __|   |  __| | . ` | |  | |
		| |  | |  | | |____  | |____| |\  | |__| |
		|_|  |_|  |_|______| |______|_| \_|_____/
		"""
		_ = _logging(message=message, filepath=aux_output_location, verbose=verbose, append=True)

		return initialfit_result, bootstrap_results


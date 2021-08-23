#%cd "C:/Users/shbhu/Documents/Git/Py_Y_X_working/PyCode"
import numpy as np
from multiprocessing import Pool,cpu_count
import os
from astropy.table import Table
import datetime

from .mle_utils_nd import MLE_fit
from .cross_validate import run_cross_validation
from .profile_likelihood import run_profile_likelihood
from .utils_nd import _save_dictionary, _logging
from .aic_nd import run_aic, run_aic_symmetric



def fit_relation(DataDict,
	SigmaLimit = 1e-3, save_path=None,
	select_deg=None, degree_max=None, k_fold=None, num_boot=100,
	cores=1, abs_tol=1e-8, verbose=2):
	"""
	Fit a Y and X relationship using a non parametric approach with beta densities

	\nINPUTS:

		save_path: Folder name (+path) to save results in.
				   Eg. save_path = '~/mrexo_working/trial_result' will create the
				   'trial_result' results folder in mrexo_working
		YSigmaLimit: The lower limit on sigma value for Y. If the sigmas are
				lower than this limit, they get changed to None. This is because,
				the Standard normal distribution blows up if the sigma values are
				too small (~1e-4). Then the distribution is no longer a convolution
				of Normal and Beta distributions, but is just a Beta distribution.
		XSigmaLimit: The lower limit on sigma value for X. If the sigmas are
				lower than this limit, they get changed to None. This is because,
				the Standard normal distribution blows up if the sigma values are
				too small (~1e-4). Then the distribution is no longer a convolution
				of Normal and Beta distributions, but is just a Beta distribution.
		select_deg: The number of degrees for the beta densities.
							if select_deg= "cv": Use cross validation to find the
								optimal number of  degrees.
							if select_deg= "aic": Use AIC minimization to find the
								optimal number of degrees.
							if select_deg= "bic": Use BIC minimization to find the
								optimal number of degrees.
							if select_deg= Integer: Use that number and skip the
								optimization process to find the number of degrees.
							NOTE: Use AIC or BIC optimization only for
								large (> 200) sample sizes.
		degree_max: Maximum degree used for cross-validation/AIC/BIC. Type:Integer.
					Default=None. If None, uses: n/np.log10(n),
					where n is the number of data points.
		k_fold: If using cross validation method, use k_fold (Integer)
				number of folds.
				Default=None.
				If None, uses: 10 folds for n > 60, 5 folds otherwise.
				Eg. k_fold=12
		num_boot: Number of bootstraps to perform. Default=100. num_boot
				must be greater than 1.
		cores: Number of cores for parallel processing. This is used in the
			   bootstrap and the cross validation. Default=1.
			   To use all the cores in the CPU,
			   cores=cpu_count() #from multiprocessing import cpu_count
		abs_tol: Absolute tolerance to be used for the numerical integration
				for product of normal and beta distribution.
				Default : 1e-8
		verbose: Integer specifying verbosity for logging.
					If 0: Will not log in the log file or print statements.
					If 1: Will write log file only.
					If 2: Will write log file and print statements.

	OUTPUTS:

		initialfit_result: Output dictionary from initial fitting without bootstrap
							using Maximum Likelihood Estimation.
							The keys in the dictionary are -
							'weights' : Weights for Beta densities from initial
								fitting w/o bootstrap.
							'aic' : Akaike Information Criterion from initial
								fitting w/o bootstrap.
							'bic' : Bayesian Information Criterion from initial
								fitting w/o bootstrap.
							'Y_points' : Sequence of Y points for initial
								fitting w/o bootstrap.
							'X_points' : Sequence of X points for initial
								fitting w/o bootstrap.
							'Y_cond_X' : Conditional distribution of Y given
								 X from initial fitting w/o bootstrap.
							'Y_cond_X_var' : Variance for the Conditional
								distribution of Y given X from initial
								fitting w/o bootstrap.
							'Y_cond_X_quantile' : Quantiles for the Conditional
								 distribution of Y given X from initial
								 fitting w/o bootstrap.
							'X_cond_Y' : Conditional distribution of X given
								 Y from initial fitting w/o bootstrap.
							'X_cond_Y_var' : Variance for the Conditional
								distribution of X given Y from initial
								fitting w/o bootstrap.
							'X_cond_Y_quantile' : Quantiles for the Conditional
								 distribution of X given Y from initial
								 fitting w/o bootstrap.
							'joint_dist' : Joint distribution of Y AND X.


		if num_boot > 2:
		bootstrap_results: Output dictionary from bootstrap run using Maximum
							Likelihood Estimation.
							'weights' : Weights for Beta densities from bootstrap run.
							'aic' : Akaike Information Criterion from bootstrap run.
							'bic' : Bayesian Information Criterion from bootstrap run.
							'Y_points' : Sequence of Y points for initial
								fitting w/o bootstrap.
							'X_points' : Sequence of X points for initial
								 fitting w/o bootstrap.
							'Y_cond_X' : Conditional distribution of Y given
								 X from bootstrap run.
							'Y_cond_X_var' : Variance for the Conditional
								 distribution of Y given X from bootstrap run.
							'Y_cond_X_quantile' : Quantiles for the Conditional
								 distribution of Y given X from bootstrap run.
							'X_cond_Y' : Conditional distribution of X given
								Y from bootstrap run.
							'X_cond_Y_var' : Variance for the Conditional
								distribution of X given Y from bootstrap run.
							'X_cond_Y_quantile' : Quantiles for the Conditional
								 distribution of X given Y from bootstrap run.


	EXAMPLE:

		# Example to fit a Y X relationship with 2 CPU cores,
			using 12 degrees, and 50 bootstraps.

		import os
		from astropy.table import Table
		import numpy as np
		from mrexo import fit_mr_relation

		pwd = '~/mrexo_working/'

		t = Table.read(os.path.join(pwd,'Cool_stars_20181109.csv'))

		# Symmetrical errorbars
		Y_sigma = (abs(t['pl_Yeerr1']) + abs(t['pl_Yeerr2']))/2
		X_sigma = (abs(t['pl_radeerr1']) + abs(t['pl_radeerr2']))/2

		# In Earth units
		Y = np.array(t['pl_Ye'])
		X = np.array(t['pl_rade'])

		# Directory to store results in
		result_dir = os.path.join(pwd,'Results_deg_12')

		##FINDME

		initialfit_result, bootstrap_results = fit_mr_relation(Y=Y,
												Y_sigma=Y_sigma,
												X=X,
												X_sigma=X_sigma,
												save_path=result_dir,
												select_deg=12,
												num_boot=50, cores=2)
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
	| |\/| || | / __| /   \| | | |
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
		print("Hasn't been implemented yet")
	elif select_deg == 'profile':
		print("Profile Likelihood is not implemented yet")
	elif select_deg == 'aic' :

		# deg_per_dim = run_aic(DataDict, degree_max, NumCandidates=20, cores=cores,
			# save_path=aux_output_location, verbose=verbose, abs_tol=abs_tol)

		deg_per_dim = run_aic_symmetric(DataDict, degree_max, NumCandidates=20, cores=cores,
			save_path=aux_output_location, verbose=verbose, abs_tol=abs_tol)

	else:
		# Use user defined value
		deg_per_dim = select_deg

	###########################################################
	## Step 2: Estimate the full model without bootstrap

	print('Running full dataset MLE before bootstrap')


	message = 'Running full dataset MLE before bootstrap\n'
	_ = _logging(message=message, filepath=aux_output_location, verbose=verbose, append=True)

	initialfit_result = MLE_fit(DataDict,  deg_per_dim=deg_per_dim,
	save_path=save_path, OutputWeightsOnly=False, CalculateJointDist=True)


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


# -*- coding: utf-8 -*-
import numpy as np
import os
from multiprocessing import Pool
from .mle_utils_nd import MLE_fit, calc_C_matrix
from .utils_nd import _logging, GiveDegreeCandidates, MakePlot, FlattenGrid
from .Optimizers import LogLikelihood
import matplotlib.pyplot as plt
import datetime

def run_cross_validation(DataDict, degree_max, k_fold=10, NumCandidates=20, 
	SymmetricDegreePerDimension=True,
	cores=1, save_path=os.path.dirname(__file__), verbose=2, abs_tol=1e-8):
	"""
	Calculate the optimal number of degrees in each dimension using the k-fold cross validation (CV) method.
	
	Parameters
	----------
	DataDict : dict
		The dictionary containing the data. See the output of :py:func:`mrexo.mle_utils_nd.InputData`.
	degree_max : int
		The maximum degree checked during degree selection.
	k_fold : int, default=10
		The number of folds to use for cross validation.
	NumCandidates : int, default=20
		The number of degree candidates to test.
	SymmetricDegreePerDimension: bool, default=True
		If True, while optimizing the number of degrees, will assume the same number of degrees in each dimension (i.e. symmetric), running through ``NumCandidates`` iterations.
		If False, while optimizing the number of degrees it can have ``NumCandidates ^ NumDimensions`` iterations. Therefore with 20 degree candidates in 2 dimensions, there will be 400 iterations to go through!
	cores : int, default=1
		The number of cores to use for parallel processing. To use all the cores in the CPU,
		   set ``cores=cpu_count()`` (requires '#from multiprocessing import cpu_count').
	save_path : str, default=os.path.dirname(__file__)
		The folder name (including path) to save results in. For example, ``save_path = '~/mrexo_working/trial_result'`` will create the 'trial_result' folder in 'mrexo_working' to contain the results.
	verbose : {0,1,2}, default=2
		Integer specifying verbosity for logging: 0 (will not log in the log file or print statements), 1 (will write log file only), or 2 (will write log file and print statements).
	abs_tol : float, default=1e-8
		The absolute tolerance to be used for the numerical integrations.
	
	Returns
	-------
	deg_choose : array[int]
		The optimal number of degrees in each dimension, chosen by k-fold CV.
	"""

	n = DataDict['DataLength']
	ndim = DataDict['ndim']

	degree_candidates = GiveDegreeCandidates(degree_max=degree_max, ndim=ndim, ncandidates=NumCandidates)
	np.savetxt(os.path.join(save_path, 'degree_candidates.txt'), degree_candidates)

	message = 'Running cross validation to estimate the number of degrees of freedom for the weights. Max candidate = {}'.format(degree_max)
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

	RandGen = np.random.choice(n, n, replace = False)
	RowSize = np.int(np.floor(n/k_fold))
	a = np.arange(n)
	DataIndicesFolded = [a[i*RowSize:(i+1)*RowSize] if i is not k_fold-1 else a[i*RowSize:] for i in range(k_fold) ]

	if not SymmetricDegreePerDimension:
		FlattenedDegrees = FlattenGrid(Inputs=[degree_candidates][0], ndim=ndim)
		FlattenedDegreeIndices = FlattenGrid(Inputs=[np.arange(NumCandidates)]*ndim, ndim=ndim)
	else:
		FlattenedDegrees = np.reshape(np.repeat(degree_candidates[0], ndim), (NumCandidates,ndim))
		FlattenedDegreeIndices = np.reshape(np.repeat(np.arange(NumCandidates), ndim), (NumCandidates, ndim))

	n_iter = len(FlattenedDegrees)

	## Map the inputs to the cross validation function. Then convert to numpy array and split in k_fold separate arrays
	# Iterator input to parallelize
	cv_input = ((i_fold,FlattenedDegrees[j], FlattenedDegreeIndices[j], DataIndicesFolded,n, RandGen, DataDict, 
		abs_tol, save_path, verbose) for i_fold in range(k_fold) for j in range(n_iter))

	# Run cross validation in parallel
	pool = Pool(processes = cores)
	cv_result = list(pool.imap(_cv_parallelize,cv_input))

	# Find the log-likelihood for each degree candidatea
	likelihood_matrix = np.split(np.array(cv_result) , k_fold)
	likelihood_per_degree = np.sum(likelihood_matrix, axis=0)

	# Save likelihood file
	np.savetxt(os.path.join(save_path,'likelihood_per_degree.txt'), likelihood_per_degree)
	deg_choose = FlattenedDegrees[np.nanargmax(likelihood_per_degree)]

	message='Finished CV. Picked {} degrees by maximizing likelihood with a loglike = {}'.format({str(deg_choose)}, np.nanmax(likelihood_per_degree))
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

	if ndim == 2:
		if not SymmetricDegreePerDimension:
			fig = MakePlot(np.reshape(likelihood_per_degree, (NumCandidates, NumCandidates)), Title='LogLike', degree_candidates=degree_candidates)
			fig.savefig(os.path.join(save_path, 'LogLike_CV.png'))
	elif SymmetricDegreePerDimension:
			fig = plt.figure()
			plt.plot(degree_candidates[0], likelihood_per_degree)
			plt.xlabel("Degrees"); plt.ylabel("Log Likelihood")
			plt.axvline(deg_choose[0], linestyle='dashed', c='k')
			plt.tight_layout()
			fig.savefig(os.path.join(save_path, 'LogLike_CV.png'))


	return deg_choose


def _cv_parallelize(cv_input):
	"""
	Serves as input finction for parallelizing.


	OUTPUT:

		like_pred : Predicted log likelihood for the i-th dataset and test_degree
	"""

	i_fold, deg_per_dim, DegreeIndex, DataIndicesFolded, n, RandGen,DataDict, \
		abs_tol, save_path,  verbose = cv_input

	SplitInterval = DataIndicesFolded[i_fold]

	# Mask = Training set
	# InvertMask = Testing Set
	Mask = np.repeat(False, n)
	Mask[RandGen[SplitInterval]] = True
	InvertMask = np.invert(Mask)

	# Test dataset - sth subset.
	TestDataDict = DataDict.copy()
	for k in ['ndim_data', 'ndim_sigma', 'ndim_LSigma', 'ndim_USigma']:
		TestDataDict[k] = DataDict[k][:,Mask] 
	TestDataDict['DataLength'] = len(TestDataDict['ndim_data'][1])

	# Corresponding training dataset k-1 in size
	TrainDataDict = DataDict.copy()
	for k in ['ndim_data', 'ndim_sigma', 'ndim_LSigma', 'ndim_USigma']:
		TrainDataDict[k] = DataDict[k][:,InvertMask] 
	TrainDataDict['DataLength'] = len(TrainDataDict['ndim_data'][1])

	message='Started cross validation for {} degree check and {} th-fold at {}'.format(deg_per_dim, i_fold, datetime.datetime.now())
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

	unpadded_weight, n_log_like = MLE_fit(TrainDataDict,  deg_per_dim=np.array(deg_per_dim), 
		abs_tol=abs_tol,
		save_path=save_path,  verbose=verbose, 
		OutputWeightsOnly=True, CalculateJointDist=False)

	# C_pdf is of shape n x deg_product where deg_product = Product of (deg_i - 2) for i in ndim
	C_pdf = calc_C_matrix(TestDataDict, deg_per_dim=np.array(deg_per_dim), 
		abs_tol=abs_tol,
		save_path=save_path, 
		verbose=verbose, 
		SaveCMatrix=False)

	# Calculate the final loglikelihood
	like_pred = LogLikelihood(C_pdf, unpadded_weight, TestDataDict['DataLength'])

	message='Finished cross validation for {} degree check and {} th-fold. LogLike = {} at {}\n'.format(deg_per_dim, i_fold, like_pred, datetime.datetime.now())
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

	return like_pred

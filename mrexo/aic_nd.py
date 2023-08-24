# -*- coding: utf-8 -*-
import numpy as np
import os
from multiprocessing import Pool
from .mle_utils_nd import MLE_fit
from .utils_nd import _logging, GiveDegreeCandidates, MakePlot, FlattenGrid
import matplotlib.pyplot as plt

"""
def run_aic_symmetric(DataDict, degree_max, NumCandidates=20, cores=1,
	save_path=os.path.dirname(__file__), verbose=2):
	'''
	Symmetric version of degree optimization using the AIC method. 
	Here instead of n different degrees chosen for 'n' dimensions, each dimension
	will have the same number of degrees, i.e.

	[d, d, ...] instead of [d, d', ...]

	'''

	message = 'Choosing degree with AIC in symmetric mode'
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

	ndim = DataDict['ndim']
	n = DataDict['DataLength']

	degree_candidates = GiveDegreeCandidates(degree_max=degree_max, ndim=ndim, ncandidates=NumCandidates)

	message = 'Using AIC method to estimate the number of degrees of freedom for the weights. Max candidate = {}\n'.format(degree_candidates.max())
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)
	
	DegreeChosen = RunAIC_flattened(DataDict=DataDict, 
		degree_candidates=degree_candidates, 
		NumCandidates=NumCandidates, 
		cores=cores, save_path=save_path, verbose=verbose, 
		SymmetricDegreePerDimension=True)
	
	message = 'Using AIC to select optimum degrees as = {}\n'.format(DegreeChosen)
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)
	
	
	return DegreeChosen
"""


def RunAIC(DataDict, degree_max, NumCandidates=20,
	SymmetricDegreePerDimension=True,
	cores=1, save_path=os.path.dirname(__file__), verbose=2):
	"""
	Calculate the optimal number of degrees in each dimension using the AIC method.
	
	Parameters
	----------
	DataDict : dict
		The dictionary containing the data. See the output of :py:func:`mrexo.mle_utils_nd.InputData`.
	degree_max : int
		The maximum degree checked during degree selection.
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
	
	Returns
	-------
	DegreeChosen : array[int]
		The optimal number of degrees in each dimension, chosen by minimizing the AIC.
	"""

	ndim = DataDict['ndim']
	n = DataDict['DataLength']

	degree_candidates = GiveDegreeCandidates(degree_max=degree_max, ndim=ndim, ncandidates=NumCandidates)

	message = 'Using AIC method to estimate the number of degrees of freedom for the weights. Max candidate = {}\n'.format(degree_candidates.max())
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)
	

	DegreeChosen = RunAIC_flattened(DataDict=DataDict, 
		degree_candidates=degree_candidates, 
		NumCandidates=NumCandidates, 
		cores=cores, save_path=save_path, verbose=verbose,
		SymmetricDegreePerDimension=SymmetricDegreePerDimension)
	
	"""
	if cores==1:
			
		if ndim==2:
			DegreeChosen = _RunAIC2D(DataDict=DataDict,
				degree_candidates=degree_candidates, NumCandidates=NumCandidates, 
				save_path=save_path, verbose=verbose)
		elif ndim==3:
			DegreeChosen = _RunAIC3D(DataDict=DataDict,
				degree_candidates=degree_candidates, NumCandidates=NumCandidates, 
				save_path=save_path, verbose=verbose)
		elif ndim==4:
			DegreeChosen = _RunAIC4D(DataDict=DataDict,
				degree_candidates=degree_candidates, NumCandidates=NumCandidates, 
				save_path=save_path, verbose=verbose)
	else:
		DegreeChosen = RunAIC_flattened(DataDict=DataDict, 
			degree_candidates=degree_candidates, 
			NumCandidates=NumCandidates, 
			cores=cores, save_path=save_path, verbose=verbose)
	"""
	
	message = 'Using AIC to select optimum degrees as = {}\n'.format(DegreeChosen)
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)
	
	
	return DegreeChosen


def _RunAIC2D(DataDict, degree_candidates, NumCandidates,
	save_path, verbose):
	"""
	Calculate the optimal number of degrees using the AIC method in 2D. Previously called by :py:func:`RunAIC`.
	"""
	
	n = DataDict['DataLength']
	ndim = DataDict['ndim']

	AIC = np.zeros(([NumCandidates]*ndim))
	FI = np.zeros(np.shape(AIC))

	loglike = np.zeros(np.shape(AIC))
	DegProduct = np.zeros(np.shape(AIC))
	NonZero = np.zeros(np.shape(AIC))
	Threshold = np.zeros(np.shape(AIC))
	
	for i in range(0, NumCandidates):
		message = "Running AIC: {}/{}".format(i, NumCandidates)
		_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)
	
		for j in range(0, NumCandidates):
			
		
			deg_per_dim = [degree_candidates[0][i], degree_candidates[1][j]]
			deg_product = np.product(deg_per_dim)
			
			output = MLE_fit(DataDict,  deg_per_dim=deg_per_dim,
				save_path=save_path, OutputWeightsOnly=False, CalculateJointDist=False, verbose=1)
				
			Weights = output['Weights']
				
			loglike[i,j] = output['loglike']
			DegProduct[i,j] = deg_product
			AIC[i,j] = output['aic']
			# FI[i,j] = output['fi']

			NonZero[i,j] = len(np.nonzero(Weights)[0])
			Threshold[i,j] = len(Weights[Weights > 1e-8])
		
	MinAICIndexFlat = np.argmin(AIC)
	MinAICIndex = np.unravel_index(MinAICIndexFlat, np.shape(AIC))
	DegreeChosen = np.array([degree_candidates[i][MinAICIndex[i]] for i in range(ndim)], dtype=int)
			
	np.savetxt(os.path.join(save_path, 'degree_candidates.txt'), degree_candidates)
	np.save(os.path.join(save_path, 'AIC.npy'), AIC)
	np.save(os.path.join(save_path, 'loglike.npy'), loglike)
	# np.save(os.path.join(save_path, 'FI.npy'), FI)
	np.save(os.path.join(save_path, 'DegProduct.npy'), DegProduct)
	np.save(os.path.join(save_path, 'NonZero.npy'), NonZero)
	
			
	fig = MakePlot(loglike, Title='loglike', degree_candidates=degree_candidates)
	fig.savefig(os.path.join(save_path, 'loglike.png'))
	
	fig = MakePlot(AIC, Title='AIC', degree_candidates=degree_candidates)
	fig.savefig(os.path.join(save_path, 'AIC.png'))

	fig = MakePlot(Threshold, Title='Threshold', degree_candidates=degree_candidates)
	fig.savefig(os.path.join(save_path, 'FI.png'))

	fig = MakePlot(DegProduct, Title='DegProduct', degree_candidates=degree_candidates)
	fig.savefig(os.path.join(save_path, 'DegProduct.png'))

	fig = MakePlot(2*(DegProduct/n - loglike), Title="AIC = 2*(DegProduct/n - LogLike)", degree_candidates=degree_candidates)
	fig.savefig(os.path.join(save_path, 'DegProducts_n_AIC.png'))

	fig = MakePlot(2*(NonZero/n - loglike), Title="AIC = 2*(NonZero/n - LogLike)", degree_candidates=degree_candidates)
	fig.savefig(os.path.join(save_path, 'NonZero_n_AIC.png'))

	# fig = MakePlot(2*(FI - loglike), Title="AIC = 2*(FI - LogLike)", degree_candidates=degree_candidates)
	# fig.savefig(os.path.join(save_path, 'FI_AIC.png'))

	# fig = MakePlot(2*(FI/n - loglike), Title="AIC = 2*(FI/n - LogLike)", degree_candidates=degree_candidates)
	# fig.savefig(os.path.join(save_path, 'FI_n_AIC.png'))
	
	return DegreeChosen
	
def _RunAIC3D(DataDict, degree_candidates, NumCandidates, save_path, verbose):
	"""
	Calculate the optimal number of degrees using the AIC method in 3D. Previously called by :py:func:`RunAIC`.
	"""
	
	n = DataDict['DataLength']
	ndim = DataDict['ndim']

	AIC = np.zeros(([NumCandidates]*ndim))
	FI = np.zeros(np.shape(AIC))

	loglike = np.zeros(np.shape(AIC))
	DegProduct = np.zeros(np.shape(AIC))
	NonZero = np.zeros(np.shape(AIC))
	Threshold = np.zeros(np.shape(AIC))
	
	for i in range(0, NumCandidates):
		for j in range(0, NumCandidates):
			message = "Running AIC:" + str(i)+','+str(j)
			_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)
	
			for k in range(0, NumCandidates):
				
				deg_per_dim = [degree_candidates[0][i], degree_candidates[1][j], degree_candidates[2][k]]
				deg_product = np.product(deg_per_dim)
				
				output = MLE_fit(DataDict,  deg_per_dim=deg_per_dim,
					save_path=save_path, OutputWeightsOnly=False, CalculateJointDist=False, verbose=1)
					
				Weights = output['Weights']
					
				loglike[i,j,k] = output['loglike']
				DegProduct[i,j,k] = deg_product
				AIC[i,j,k] = output['aic']
				FI[i,j,k] = output['fi']

				NonZero[i,j,k] = len(np.nonzero(Weights)[0])
				Threshold[i,j,k] = len(Weights[Weights > 1e-8])
		
	MinAICIndexFlat = np.argmin(AIC)
	MinAICIndex = np.unravel_index(MinAICIndexFlat, np.shape(AIC))
	DegreeChosen = np.array([degree_candidates[i][MinAICIndex[i]] for i in range(ndim)], dtype=int)

	np.savetxt(os.path.join(save_path, 'degree_candidates.txt'), degree_candidates)
	np.save(os.path.join(save_path, 'AIC.npy'), AIC)
	np.save(os.path.join(save_path, 'loglike.npy'), loglike)
	np.save(os.path.join(save_path, 'FI.npy'), FI)
	np.save(os.path.join(save_path, 'DegProduct.npy'), DegProduct)
	np.save(os.path.join(save_path, 'NonZero.npy'), NonZero)
	
			
	return DegreeChosen
	

def _RunAIC4D(DataDict, degree_candidates, NumCandidates, save_path, verbose):
	"""
	Calculate the optimal number of degrees using the AIC method in 4D. Previously called by :py:func:`RunAIC`.
	"""
	
	n = DataDict['DataLength']
	ndim = DataDict['ndim']

	AIC = np.zeros(([NumCandidates]*ndim))
	FI = np.zeros(np.shape(AIC))

	loglike = np.zeros(np.shape(AIC))
	DegProduct = np.zeros(np.shape(AIC))
	NonZero = np.zeros(np.shape(AIC))
	Threshold = np.zeros(np.shape(AIC))
	
	for i in range(0, NumCandidates):
		for j in range(0, NumCandidates):
			for k in range(0, NumCandidates):
				message = "Running AIC: {}{}{}".format(i,j,k)
				_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)
	
				for l in range(0, NumCandidates):
						
					deg_per_dim = [degree_candidates[0][i], degree_candidates[1][j], degree_candidates[2][k], degree_candidates[3][l]]
					deg_product = np.product(deg_per_dim)
					
					output = MLE_fit(DataDict,  deg_per_dim=deg_per_dim,
						save_path=save_path, OutputWeightsOnly=False, CalculateJointDist=False, verbose=1)
						
					Weights = output['Weights']
						
					loglike[i,j,k,l] = output['loglike']
					DegProduct[i,j,k,l] = deg_product
					AIC[i,j,k,l] = output['aic']
					FI[i,j,k,l] = output['fi']

					NonZero[i,j,k,l] = len(np.nonzero(Weights)[0])
					Threshold[i,j,k,l] = len(Weights[Weights > 1e-8])
			
	MinAICIndexFlat = np.argmin(AIC)
	MinAICIndex = np.unravel_index(MinAICIndexFlat, np.shape(AIC))
	DegreeChosen = np.array([degree_candidates[i][MinAICIndex[i]] for i in range(ndim)], dtype=int)

	np.savetxt(os.path.join(save_path, 'degree_candidates.txt'), degree_candidates)
	np.save(os.path.join(save_path, 'AIC.npy'), AIC)
	np.save(os.path.join(save_path, 'loglike.npy'), loglike)
	np.save(os.path.join(save_path, 'FI.npy'), FI)
	np.save(os.path.join(save_path, 'DegProduct.npy'), DegProduct)
	np.save(os.path.join(save_path, 'NonZero.npy'), NonZero)
	
			
	return DegreeChosen


def RunAIC_flattened(DataDict, degree_candidates, NumCandidates, cores, save_path, verbose, 
	SymmetricDegreePerDimension=False):
	"""
	Called by :py:func:`RunAIC` to calculate the optimal number of degrees in each dimension using the AIC method.
	
	Parameters
	----------
	DataDict : dict
		The dictionary containing the data. See the output of :py:func:`mrexo.mle_utils_nd.InputData`.
	degree_candidates : array[int]
		The vector of degree candidates (e.g., as given by :py:func:`mrexo.utils_nd.GiveDegreeCandidates`).
	NumCandidates : int, default=20
		The number of degree candidates to test.
	cores : int
		The number of cores to use for parallel processing. To use all the cores in the CPU,
		   set ``cores=cpu_count()`` (requires '#from multiprocessing import cpu_count').
	save_path : str, default=os.path.dirname(__file__)
		The folder name (including path) to save results in. For example, ``save_path = '~/mrexo_working/trial_result'`` will create the 'trial_result' folder in 'mrexo_working' to contain the results.
	verbose : {0,1,2}, default=2
		Integer specifying verbosity for logging: 0 (will not log in the log file or print statements), 1 (will write log file only), or 2 (will write log file and print statements).
	SymmetricDegreePerDimension: bool, default=False
		If True, while optimizing the number of degrees, will assume the same number of degrees in each dimension (i.e. symmetric), running through ``NumCandidates`` iterations.
		If False, while optimizing the number of degrees it can have ``NumCandidates ^ NumDimensions`` iterations. Therefore with 20 degree candidates in 2 dimensions, there will be 400 iterations to go through!
	
	Returns
	-------
	DegreeChosen : array[int]
		The optimal number of degrees in each dimension, chosen by minimizing the AIC.
	"""

	n = DataDict['DataLength']
	ndim = DataDict['ndim']

	if not SymmetricDegreePerDimension:
		FlattenedDegrees = FlattenGrid(Inputs=[degree_candidates][0], ndim=ndim)
		FlattenedDegreeIndices = FlattenGrid(Inputs=[np.arange(NumCandidates)]*ndim, ndim=ndim)
	else:
		FlattenedDegrees = np.reshape(np.repeat(degree_candidates[0], ndim), (NumCandidates,ndim))
		FlattenedDegreeIndices = np.reshape(np.repeat(np.arange(NumCandidates), ndim), (NumCandidates, ndim))
	
	n_iter = len(FlattenedDegrees)
	
	inputs_aicpool = ((DataDict, FlattenedDegrees[i], FlattenedDegreeIndices[i], save_path, verbose) for i in range(n_iter))

	AICgrid = np.zeros(([NumCandidates]*ndim))
	AICgrid[:] = np.nan
	AIC_FIgrid = np.zeros(([NumCandidates]*ndim))
	AIC_FIgrid[:] = np.nan
	LoglikeGrid = np.zeros(([NumCandidates]*ndim))
	NonZeroGrid = np.zeros(([NumCandidates]*ndim))
	ThresholdGrid8 = np.zeros(([NumCandidates]*ndim))
	ESSGrid = np.zeros(([NumCandidates]*ndim))

	if cores > 1:
		# Parallelize the AIC
		pool = Pool(processes=cores, initializer=np.random.seed)
		aic_results = list(pool.imap(_AIC_MLE, inputs_aicpool))

		
		AIC = np.array([x['aic'] for x in aic_results])
		AIC_FI = np.array([x['aic_fi'] for x in aic_results])

		Index = np.array([x['index'] for x in aic_results])
		LogLike = np.array([x['loglike'] for x in aic_results])
		Weights = [x['Weights'] for x in aic_results]
		ESS = [x['EffectiveDOF'] for x in aic_results]
		
		for i in range(n_iter):
			
			AICgrid[tuple(Index[i])] = AIC[i]
			AIC_FIgrid[tuple(Index[i])] = AIC_FI[i]

			w = Weights[i]
			
			NonZeroGrid[tuple(Index[i])] = len(np.nonzero(w)[0])
			LoglikeGrid[tuple(Index[i])] = LogLike[i]
			ThresholdGrid8[tuple(Index[i])] = len(w[w>1e-8])
			ESSGrid[tuple(Index[i])] = ESS[i]

	else:
		
		i  = 0
		for inputs in inputs_aicpool:
			output = _AIC_MLE(inputs)
			Index = output['index']
			AIC = output['aic']
			AIC_FI = output['aic_fi']
			LogLike = output['loglike']
			Weights = output['Weights']
			ESS = output['EffectiveDOF']
			
			AICgrid[tuple(Index)] = AIC
			AIC_FIgrid[tuple(Index)] = AIC_FI

			w = Weights
			
			NonZeroGrid[Index] = len(np.nonzero(w)[0])
			LoglikeGrid[Index] = LogLike
			ThresholdGrid8[Index] = len(w[w>1e-8])
			ESSGrid[Index] = ESS
			i += 1

		
	MinAICIndexFlat = np.nanargmin(AICgrid)
	MinAICIndex = np.unravel_index(MinAICIndexFlat, np.shape(AICgrid))
	DegreeChosen = np.array([degree_candidates[i][MinAICIndex[i]] for i in range(ndim)], dtype=int)
	
	np.savetxt(os.path.join(save_path, 'degree_candidates.txt'), degree_candidates)
	np.save(os.path.join(save_path, 'AIC.npy'), AICgrid)
	np.save(os.path.join(save_path, 'AIC_FI.npy'), AIC_FIgrid)
	np.save(os.path.join(save_path, 'loglike.npy'), LoglikeGrid)
	np.save(os.path.join(save_path, 'NonZero.npy'), NonZeroGrid)
	np.save(os.path.join(save_path, 'Weights_AIC.npy'), Weights)
	np.save(os.path.join(save_path, 'ThresholdGrid8.npy'), ThresholdGrid8)
	np.save(os.path.join(save_path, 'EffectiveSampleSize.npy'), ESSGrid)


	if not SymmetricDegreePerDimension:
		# Make 2D plots in the case of asymmetric degrees for each dimension
		if ndim==2:
			fig = MakePlot(LoglikeGrid, Title='loglike', degree_candidates=degree_candidates)
			fig.savefig(os.path.join(save_path, 'loglike.png'))
			
			fig = MakePlot(AICgrid, Title='AIC', degree_candidates=degree_candidates)
			fig.savefig(os.path.join(save_path, 'AIC.png'))

			fig = MakePlot(AIC_FIgrid, Title='AIC_FI', degree_candidates=degree_candidates)
			fig.savefig(os.path.join(save_path, 'AIC_FI.png'))

			fig = MakePlot(NonZeroGrid, Title='Nonzero', degree_candidates=degree_candidates)
			fig.savefig(os.path.join(save_path, 'NonZero.png'))

			fig = MakePlot(ThresholdGrid8, Title='ThresholdGrid8', degree_candidates=degree_candidates)
			fig.savefig(os.path.join(save_path, 'ThresholdGrid8.png'))

			fig = MakePlot(2*(ThresholdGrid8/n - LoglikeGrid), Title="AIC = 2*(ThresholdGrid8/n - LogLike)", degree_candidates=degree_candidates)
			fig.savefig(os.path.join(save_path, 'ThresholdGrid8_n_AIC.png'))

			fig = MakePlot(2*(NonZeroGrid/n - LoglikeGrid), Title="AIC = 2*(NonZero/n - LogLike)", degree_candidates=degree_candidates)
			fig.savefig(os.path.join(save_path, 'NonZero_n_AIC.png'))

	else:
		fig = plt.figure()
		plt.plot(degree_candidates[0], [AICgrid[tuple([i]*ndim)] for i in range(NumCandidates)]) #for ndim=2, this is equivalent to np.diag(AICgrid)
		plt.xlabel("Degrees"); plt.ylabel("AIC")
		plt.axvline(DegreeChosen[0], linestyle='dashed', c='k')
		plt.tight_layout()
		fig.savefig(os.path.join(save_path, 'AIC.png'))
		plt.close("all")

	return DegreeChosen
	

def _AIC_MLE(inputs):
	
	DataDict, deg_per_dim, index, save_path, verbose = inputs
	
	message = "Running degrees = {}\n".format(deg_per_dim)
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

	outputs = MLE_fit(DataDict,  deg_per_dim=deg_per_dim,
		save_path=save_path, OutputWeightsOnly=False, CalculateJointDist=False, verbose=verbose)
		
	outputs['index'] = index
	
	return outputs
	

	
		

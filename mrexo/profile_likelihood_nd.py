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

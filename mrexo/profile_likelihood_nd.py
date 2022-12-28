# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd
from multiprocessing import Pool
from .mle_utils_nd import MLE_fit, calc_C_matrix
from .utils_nd import _logging, GiveDegreeCandidates

"""
def run_profile_likelihood(Y, X, Y_sigma, X_sigma, Y_bounds, X_bounds,
                        X_char='x', Y_char='y',
                        degree_max=None, degree_candidates=None,
                        cores=1, save_path=os.path.dirname(__file__), verbose=2, abs_tol=1e-8):
    """
    """

    n = len(Y)
"""


def run_profile_likelihood(DataDict, degree_max, cores=1,
	save_path=os.path.dirname(__file__), verbose=2, abs_tol=1e-8):
	"""
	
	degree_max = A np.array with number of elements equal to number of degrees, 
		with each element corresponding to the maximum degree for each dimension.
	"""

	ndim = DataDict['ndim']
	n = DataDict['DataLength']
	"""
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
	"""

	degree_candidates = GiveDegreeCandidates(degree_max, n, ndim)

    message = 'Running profile likelihood method to estimate the number of degrees of freedom for the weights. Max candidate = {}\n'.format(degree_candidates.max())
    _ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

    # For profile likelihood to pick the optimum number of degrees,
    # Define tolerance to 0.01 or 1/n whichever is larger.
    # Do not want arbitrarily small tolerances for large sample sizes, and
    # for small samples 1% might never be met.
    # logliketolerance = max(0.01, 1/n)

    if cores>1:
        message = "Running profile likelihood in parallel"
    else:
        message = "Running profile likelihood in serial"
        _ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)
        
        loglike = np.zeros((len(degree_candidates[0]), len(degree_candidates[1])))

		for i in range(0, len(degree_candidates[0])):
			for j in range(0, len(degree_candidates[1])):
					_, loglike[i,j] = MLE_fit(DataDict,  deg_per_dim=[degree_candidates[0][i], degree_candidates[1][j]],
			save_path=save_path, OutputWeightsOnly=True, CalculateJointDist=False)

	Gradient = np.gradient(loglike, *degree_candidates)
	TotalDerivative = (Gradient[0]**2 + Gradient[1]**2)

	plt.imshow(loglike, origin='lower', interpolation='bicubic')
	plt.colorbar()
	plt.title(degree_max)
	plt.show(block=False)

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

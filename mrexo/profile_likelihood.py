# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd
from multiprocessing import Pool
from .mle_utils import MLE_fit, calc_C_matrix
from .utils import _save_dictionary, _logging


def RunProfileLikelihood(Y, X, Y_sigma, X_sigma, Y_bounds, X_bounds,
                        X_char='x', Y_char='y',
                        degree_max=None, degree_candidates=None,
                        cores=1, save_path=os.path.dirname(__file__), logliketolerance=1e-2, verbose=2, abs_tol=1e-8):
    """
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
    """

    n = len(Y)

    if degree_candidates is None:
        if degree_max is None:
            degree_candidates = (np.floor(n**np.arange(0.4, 0.76, 0.05))).astype(int)
        else:
            degree_candidates = (np.floor(n**np.arange(0.4, np.log(degree_max)/np.log(n), 0.05))).astype(int)

    message = 'Running profile likelihood method to estimate the number of degrees of freedom for the weights. Max candidate = {}\n'.format(degree_candidates.max())
    _ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

    """
    pl_input = (Y, X, Y_sigma, X_sigma, Y_bounds, X_bounds, Y_char, X_char,
                degree_candidates, abs_tol, save_path, verbose)

    # Run cross validation in parallel
    pool = Pool(processes = cores)
    pl_result = list(pool.imap(_parallelize_profilelikelihood, pl_input))
    likelihood_per_degree = np.array(pl_result)
    """

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
            print("Threshold not reached for maximum degree="+str(degree_candidates.max()))
            break
        else:
            t+=1

    df = pd.DataFrame({"degree_candidates":degree_candidates, "LogLikelihood":loglike, "FractionalChange":FractionalChange})
    df.to_csv(os.path.join(save_path, 'likelihood_per_degree.csv'), index=False)
    # np.savetxt(os.path.join(save_path,'likelihood_per_degree.txt'),np.array([degree_candidates, loglike, FractionalChange]))
    message='Finished Profile Likelihood. Picked {} degrees.'.format({deg_choose})
    _ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)




    return deg_choose


def _parallelize_profilelikelihood(pl_input):
    Y, X, Y_sigma, X_sigma, Y_bounds, X_bounds, Y_char, X_char, deg, abs_tol, save_path, verbose = pl_input[0]


    _, loglike = MLE_fit(Y=Y, X=X, Y_sigma=Y_sigma, X_sigma=X_sigma,
            Y_bounds=Y_bounds, X_bounds=X_bounds, Y_char=Y_char, X_char=X_char,
            deg=deg, abs_tol=abs_tol, save_path=save_path, output_weights_only=True, verbose=verbose)
    return loglike

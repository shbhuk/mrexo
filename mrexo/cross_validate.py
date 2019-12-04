# -*- coding: utf-8 -*-
import numpy as np
import os
from multiprocessing import Pool
from .mle_utils_beta import MLE_fit, calc_C_matrix
from .utils import _save_dictionary, _logging


def run_cross_validation(Y, X, Y_sigma, X_sigma, Y_bounds, X_bounds,
                        X_char='x', Y_char='y',
                        degree_max=60, k_fold=10, degree_candidates=None,
                        cores=1, save_path=os.path.dirname(__file__), abs_tol=1e-8, verbose=2):
    """
    We use k-fold cross validation to choose the optimal number of degrees from a set of input candidate degree values.
    To conduct the k-fold cross validation, we separate the dataset randomly into k disjoint subsets with equal
    sizes. Then we leave out the s-th subset, denoted by Ds (s = 1, . . . , k) and use the remaining k-1 subsets of data
    to estimate the weights of the beta densities (w_s). Repeating this for each s-th subset Ds results in k estimated sets of weights (w_s).
    We plug in each weight (w_s) along with the corresponding data, to obtain an estimated value for the log-likelihood.
    This log likelihood is maximized to pick the optimum number of degrees.

    Refer to Ning et al. 2018 Sec 2.2

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
    if degree_candidates == None:
        degree_candidates = np.linspace(5, degree_max, 10, dtype = int)

    n = len(Y)

    message = 'Running cross validation to estimate the number of degrees of freedom for the weights. Max candidate = {}'.format(degree_max)
    _ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

    rand_gen = np.random.choice(n, n, replace = False)
    row_size = np.int(np.floor(n/k_fold))
    a = np.arange(n)
    indices_folded = [a[i*row_size:(i+1)*row_size] if i is not k_fold-1 else a[i*row_size:] for i in range(k_fold) ]

    ## Map the inputs to the cross validation function. Then convert to numpy array and split in k_fold separate arrays
    # Iterator input to parallelize
    cv_input = ((i,j, indices_folded,n, rand_gen, Y, X, X_sigma, Y_sigma,
     abs_tol, save_path, Y_bounds, X_bounds, Y_char, X_char, verbose) for i in range(k_fold) for j in degree_candidates)

    # Run cross validation in parallel
    pool = Pool(processes = cores)
    cv_result = list(pool.imap(_cv_parallelize,cv_input))

    # Find the log-likelihood for each degree candidatea
    likelihood_matrix = np.split(np.array(cv_result) , k_fold)
    likelihood_per_degree = np.sum(likelihood_matrix, axis=0)

    # Save likelihood file
    np.savetxt(os.path.join(save_path,'likelihood_per_degree.txt'),np.array([degree_candidates,likelihood_per_degree]))
    deg_choose = degree_candidates[np.argmax(likelihood_per_degree)]

    message='Finished CV. Picked {} degrees by maximizing likelihood'.format({deg_choose})
    _ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

    return deg_choose


def _cv_parallelize(cv_input):
    """
    Serves as input finction for parallelizing.

    \nINPUTS:

        cv_input : Tuple with following components:
            i_fold : i out of k-th fold being run.
            test_degree: The degree candidate that is being tested.
            indices_folded: The indices for the i-th fold.
            n: Number of total data points.
            rand_gen: List of randomized indices (sampled without replacement) used to extract i-th fold datapoints.
            Y: Numpy array of Y measurements. In LINEAR SCALE.
            X: Numpy array of X measurements. In LINEAR SCALE.
            Y_sigma: Numpy array of Y uncertainties. Assumes symmetrical uncertainty. In LINEAR SCALE.
            X_sigma: Numpy array of X uncertainties. Assumes symmetrical uncertainty. In LINEAR SCALE.
            abs_tol : Absolute tolerance to be used for the numerical integration for product of normal and beta distribution.
                    Default : 1e-8
            save_path: Location of folder within results for auxiliary output files
            Y_bounds: Bounds for the Y. Log10
            X_bounds: Bounds for the X. Log10

    OUTPUT:

        like_pred : Predicted log likelihood for the i-th dataset and test_degree
    """
    i_fold, test_degree, indices_folded, n, rand_gen, Y, X, X_sigma, Y_sigma, abs_tol,\
        save_path, Y_bounds, X_bounds, Y_char, X_char, verbose = cv_input
    split_interval = indices_folded[i_fold]

    mask = np.repeat(False, n)
    mask[rand_gen[split_interval]] = True
    invert_mask = np.invert(mask)

    # Test dataset - sth subset.
    test_X = X[mask]
    test_Y = Y[mask]
    test_X_sigma = X_sigma[mask]
    test_Y_sigma = Y_sigma[mask]

    # Corresponding training dataset k-1 in size
    train_X = X[invert_mask]
    train_Y = Y[invert_mask]
    train_X_sigma = X_sigma[invert_mask]
    train_Y_sigma = Y_sigma[invert_mask]

    message='Running cross validation for {} degree check and {} th-fold\n'.format(test_degree, i_fold)
    _ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)

    # Calculate the optimum weights using MLE for a given input test_degree
    weights = MLE_fit(Y=train_Y, X=train_X, Y_sigma=train_Y_sigma, X_sigma=train_X_sigma,
            Y_bounds=Y_bounds, X_bounds=X_bounds, Y_char=Y_char, X_char=X_char,
            deg=test_degree, abs_tol=abs_tol, save_path=save_path, output_weights_only=True, verbose=verbose)

    size_test = np.size(test_X)

    # Specify the bounds
    Y_max = Y_bounds[1]
    Y_min = Y_bounds[0]
    X_max = X_bounds[1]
    X_min = X_bounds[0]

    print(size_test, len(Y_bounds))

    # Integrate the product of the normal and beta distribution for Y and X and then take the Kronecker product
    C_pdf = calc_C_matrix(size_test, test_degree, test_Y, test_Y_sigma, Y_max, Y_min, test_X,
                        test_X_sigma, X_max, X_min,  abs_tol, save_path, Log=True, verbose=verbose)

    # Calculate the final loglikelihood
    like_pred =  np.sum(np.log(np.matmul(weights,C_pdf)))

    return like_pred

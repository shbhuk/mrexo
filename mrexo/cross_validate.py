# -*- coding: utf-8 -*-
import numpy as np
import os
from multiprocessing import Pool
from .mle_utils import MLE_fit, calc_C_matrix

def run_cross_validation(Mass, Radius, Mass_sigma, Radius_sigma, Mass_bounds, Radius_bounds,
                        degree_max=60, k_fold=10, degree_candidates=None,
                        cores=1, save_path=os.path.dirname(__file__), abs_tol=1e-8):
    '''
    We use k-fold cross validation to choose the optimal number of degrees from a set of input candidate degree values.
    To conduct the k-fold cross validation, we separate the dataset randomly into k disjoint subsets with equal
    sizes. Then we leave out the s-th subset, denoted by Ds (s = 1, . . . , k) and use the remaining k-1 subsets of data
    to estimate the weights of the beta densities (w_s). Repeating this for each s-th subset Ds results in k estimated sets of weights (w_s).
    We plug in each weight (w_s) along with the corresponding data, to obtain an estimated value for the log-likelihood.
    This log likelihood is maximized to pick the optimum number of degrees.

    Refer to Ning et al. 2018 Sec 2.2

    INPUTS:
        Mass: Numpy array of mass measurements. In LINEAR SCALE.
        Radius: Numpy array of radius measurements. In LINEAR SCALE.
        Mass_sigma: Numpy array of mass uncertainties. Assumes symmetrical uncertainty. In LINEAR SCALE.
        Radius_sigma: Numpy array of radius uncertainties. Assumes symmetrical uncertainty. In LINEAR SCALE.
        Mass_bounds: Bounds for the mass. Log10
        Radius_bounds: Bounds for the radius. Log10
        degree_max: Maximum degree used for cross-validation/AIC/BIC. Type: Integer.
                    Default=None. If None, uses: n/np.log10(n), where n is the number of data points.
        k_fold: If using cross validation method, use k_fold (integer) number of folds. Default=None.
                If None, uses:
                  - 10 folds for n > 60, where n is the length of the Mass and Radius arrays.
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

    OUTPUTS:
        deg_choose - The optimum degree chosen by cross validation and MLE
    '''
    if degree_candidates == None:
        degree_candidates = np.linspace(5, degree_max, 12, dtype = int)

    n = len(Mass)

    print('Running cross validation to estimate the number of degrees of freedom for the weights. Max candidate = {}'.format(degree_max))
    rand_gen = np.random.choice(n, n, replace = False)
    row_size = np.int(np.floor(n/k_fold))
    a = np.arange(n)
    indices_folded = [a[i*row_size:(i+1)*row_size] if i is not k_fold-1 else a[i*row_size:] for i in range(k_fold) ]

    ## Map the inputs to the cross validation function. Then convert to numpy array and split in k_fold separate arrays
    # Iterator input to parallelize
    cv_input = ((i,j, indices_folded,n, rand_gen, Mass, Radius, Radius_sigma, Mass_sigma,
     abs_tol, save_path, Mass_bounds, Radius_bounds) for i in range(k_fold) for j in degree_candidates)

    # Run cross validation in parallel
    pool = Pool(processes = cores)
    cv_result = list(pool.imap(cv_parallelize,cv_input))

    # Find the log-likelihood for each degree candidatea
    likelihood_matrix = np.split(np.array(cv_result) , k_fold)
    likelihood_per_degree = np.sum(likelihood_matrix, axis=0)

    # Save likelihood file
    np.savetxt(os.path.join(save_path,'likelihood_per_degree.txt'),np.array([degree_candidates,likelihood_per_degree]))
    deg_choose = degree_candidates[np.argmax(likelihood_per_degree)]

    print('Finished CV. Picked {} degrees by maximizing likelihood'.format({deg_choose}))

    return deg_choose


def cv_parallelize(cv_input):
    '''
    Serves as input finction for parallelizing.

    INPUTS:
        cv_input : Tuple with following components:
            i_fold : i out of k-th fold being run.
            test_degree: The degree candidate that is being tested.
            indices_folded: The indices for the i-th fold.
            n: Number of total data points.
            rand_gen: List of randomized indices (sampled without replacement) used to extract i-th fold datapoints.
            Mass: Numpy array of mass measurements. In LINEAR SCALE.
            Radius: Numpy array of radius measurements. In LINEAR SCALE.
            Mass_sigma: Numpy array of mass uncertainties. Assumes symmetrical uncertainty. In LINEAR SCALE.
            Radius_sigma: Numpy array of radius uncertainties. Assumes symmetrical uncertainty. In LINEAR SCALE.
            abs_tol : Absolute tolerance to be used for the numerical integration for product of normal and beta distribution.
                    Default : 1e-8
            save_path: Location of folder within results for auxiliary output files
            Mass_bounds: Bounds for the mass. Log10
            Radius_bounds: Bounds for the radius. Log10

    OUTPUT:
        like_pred : Predicted log likelihood for the i-th dataset and test_degree
    '''
    i_fold, test_degree, indices_folded, n, rand_gen, Mass, Radius, Radius_sigma, Mass_sigma, abs_tol, save_path, Mass_bounds, Radius_bounds = cv_input
    split_interval = indices_folded[i_fold]

    mask = np.repeat(False, n)
    mask[rand_gen[split_interval]] = True
    invert_mask = np.invert(mask)

    # Test dataset - sth subset.
    test_Radius = Radius[mask]
    test_Mass = Mass[mask]
    test_Radius_sigma = Radius_sigma[mask]
    test_Mass_sigma = Mass_sigma[mask]

    # Corresponding training dataset k-1 in size
    train_Radius = Radius[invert_mask]
    train_Mass = Mass[invert_mask]
    train_Radius_sigma = Radius_sigma[invert_mask]
    train_Mass_sigma = Mass_sigma[invert_mask]

    with open(os.path.join(save_path,'log_file.txt'),'a') as f:
       f.write('Running cross validation for {} degree check and {} th-fold\n'.format(test_degree, i_fold))

    # Calculate the optimum weights using MLE for a given input test_degree
    weights = MLE_fit(Mass=train_Mass, Radius=train_Radius, Mass_sigma=train_Mass_sigma, Radius_sigma=train_Radius_sigma, Mass_bounds=Mass_bounds,
            Radius_bounds=Radius_bounds, deg=test_degree, abs_tol=abs_tol, save_path=save_path, output_weights_only=True)

    size_test = np.size(test_Radius)

    # Specify the bounds
    Mass_max = Mass_bounds[1]
    Mass_min = Mass_bounds[0]
    Radius_max = Radius_bounds[1]
    Radius_min = Radius_bounds[0]

    print(size_test, len(Mass_bounds))

    # Integrate the product of the normal and beta distribution for mass and radius and then take the Kronecker product
    C_pdf = calc_C_matrix(size_test, test_degree, test_Mass, test_Mass_sigma, Mass_max, Mass_min, test_Radius, test_Radius_sigma, Radius_max, Radius_min,  abs_tol, save_path, Log=True)

    # Calculate the final loglikelihood
    like_pred =  np.sum(np.log(np.matmul(weights,C_pdf)))

    return like_pred

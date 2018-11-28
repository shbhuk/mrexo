import numpy as np
import os
from multiprocessing import Pool

from .mle_utils import MLE_fit, calc_C_matrix




def run_cross_validation(Mass, Radius, Mass_sigma, Radius_sigma, Mass_bounds, Radius_bounds, 
                        degree_max = 60, k_fold = 10, degree_candidate = None,
                        cores = 1, save_path = os.path.dirname(__file__), abs_tol = 1e-10):
    '''
    Mass: Mass measurements
    Radius : Radius measurements
    Mass_sigma: measurement errors for the data, if no measuremnet error,
                it is NULL
    Radius_sigma
    Mass_max: the upper bound for mass. Default = None
    Mass_min: the lower bound for mass. Default = None
    Radius_max: the upper bound for radius. Default = None
    Radius_min: the upper bound for radius. Default = None

    degree_max: the maximum degree used for cross-validation/AIC/BIC. Default = 60. Suggested value = n/log(n)
    k_fold: number of fold used for cross validation. Default = 10
    degree_candidate : Integer vector containing degrees to run cross validation check for. Default is None.
                    If None, defaults to
    cores: this program uses parallel computing for bootstrap. Default = 1
    save_path : The save_path for the log file
    abs_tol : Defined for integration in MLE_fit()

    '''
    if degree_candidate == None:
        degree_candidate = np.linspace(5, degree_max, 12, dtype = int)

    n = len(Mass)

    print('Running cross validation to estimate the number of degrees of freedom for the weights. Max candidate = {}'.format(degree_max))
    rand_gen = np.random.choice(n, n, replace = False)
    row_size = np.int(np.floor(n/k_fold))
    a = np.arange(n)
    indices_folded = [a[i*row_size:(i+1)*row_size] if i is not k_fold-1 else a[i*row_size:] for i in range(k_fold) ]

    cv_input = ((i,j, indices_folded,n, rand_gen, Mass, Radius, Radius_sigma, Mass_sigma,
     abs_tol, save_path, Mass_bounds, Radius_bounds) for i in range(k_fold)for j in degree_candidate)

    pool = Pool(processes = cores)

    # Map the inputs to the cross validation function. Then convert to numpy array and split in k_fold separate arrays
    cv_result = list(pool.imap(cv_parallelize,cv_input))
    likelihood_matrix = np.split(np.array(cv_result) , k_fold)
    likelihood_per_degree = np.sum(likelihood_matrix, axis = 0)

    print(likelihood_per_degree)
    np.savetxt(os.path.join(save_path,'likelihood_per_degree.txt'),np.array([degree_candidate,likelihood_per_degree]))
    deg_choose = degree_candidate[np.argmax(likelihood_per_degree)]

    print('Finished CV. Picked {} degrees by maximizing likelihood'.format({deg_choose}))

    return (deg_choose)


def cv_parallelize(cv_input):

    '''
    Function to run in parallel the cross validation routine.
    train_data = Input training data
    bounds = Vector containing four elements. Upper and lower bound for Mass. Upper and lower bound for Radius resp.
    test_radius, = Input radius data
    test_mass, = Input mass data
    deg = Degree used in Bernstein Polynomials
    train_data_sg = Input training data measurement errors
    test_radius_sigma = Input test radius measurement errors
    test_mass_sigma = Input test mass measurement errors
    Log = Whether the data is in log units
    abs_tol = Absolute tolerance
    save_path : The save_path for the log file
    '''
    i_fold, test_degree, indices_folded, n, rand_gen, Mass, Radius, Radius_sigma, Mass_sigma, abs_tol, save_path, Mass_bounds, Radius_bounds = cv_input
    split_interval = indices_folded[i_fold]

    mask = np.repeat(False, n)
    mask[rand_gen[split_interval]] = True
    invert_mask = np.invert(mask)

    test_Radius = Radius[mask]
    test_Mass = Mass[mask]
    test_Radius_sigma = Radius_sigma[mask]
    test_Mass_sigma = Mass_sigma[mask]

    train_Radius = Radius[invert_mask]
    train_Mass = Mass[invert_mask]
    train_Radius_sigma = Radius_sigma[invert_mask]
    train_Mass_sigma = Mass_sigma[invert_mask]

    with open(os.path.join(save_path,'log_file.txt'),'a') as f:
       f.write('Running cross validation for {} degree check and {} th-fold'.format(test_degree, i_fold))

    weights = MLE_fit(Mass = train_Mass, Radius = train_Radius, Mass_sigma = train_Mass_sigma, Radius_sigma = train_Radius_sigma, Mass_bounds = Mass_bounds,
            Radius_bounds = Radius_bounds, deg = test_degree, abs_tol = abs_tol, save_path = save_path, output_weights_only = True)

    size_test = np.size(test_Radius)

    # specify the bounds
    Mass_max = Mass_bounds[1]
    Mass_min = Mass_bounds[0]
    Radius_max = Radius_bounds[1]
    Radius_min = Radius_bounds[0]

    # calculate cdf and pdf of M and R for each term
    # the first and last term is set to 0 to avoid boundary effects
    # so we only need to calculate 2:(deg^2-1) terms

    C_pdf = calc_C_matrix(size_test, test_degree, test_Mass, test_Mass_sigma, Mass_max, Mass_min, test_Radius, test_Radius_sigma, Radius_max, Radius_min,  abs_tol, save_path)

    like_pred =  np.sum(np.log(np.matmul(weights,C_pdf)))

    return like_pred

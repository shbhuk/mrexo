#%cd "C:/Users/shbhu/Documents/Git/Py_Y_X_working/PyCode"
import numpy as np
from multiprocessing import Pool,cpu_count
import os
from astropy.table import Table
import datetime

from .mle_utils import MLE_fit
from .cross_validate import run_cross_validation
from .utils import _save_dictionary, _logging



def fit_xy_relation(Y, Y_sigma, X, X_sigma, save_path,
                    X_label, Y_label, X_char='x', Y_char='y',
                    Y_min=None, Y_max=None, X_min=None, X_max=None,
                    YSigmaLimit = 1e-3, XSigmaLimit = 1e-3,
                    select_deg=17, degree_max=None, k_fold=None, num_boot=100,
                    cores=1, abs_tol=1e-8, verbose=2):
    """
    Fit a Y and X relationship using a non parametric approach with beta densities

    \nINPUTS:

        Y:Numpy array of Y measurements. In LINEAR SCALE.
        Y_sigma: Numpy array of Y uncertainties. Assumes symmetrical uncer-
                    -tainty. In LINEAR SCALE.
                    If no uncertainties then use np.nan. Array must have same size
                    as measurements (Y array)
        X: Numpy array of X measurements. In LINEAR SCALE.
        X_sigma: Numpy array of X uncertainties. Assumes symmetrical
                    uncertainty. In LINEAR SCALE.
                    If no uncertainties then use np.nan. Array must have same size
                    as measurements (X array)
        save_path: Folder name (+path) to save results in.
                   Eg. save_path = '~/mrexo_working/trial_result' will create the
                   'trial_result' results folder in mrexo_working
        X_label: String label for X. Example Radius/Mass/Period
        Y_label: String label for Y. Example Radius/Mass/Period
        X_char: String alphabet (character) to depict X quantity.
            Eg 'm' for Mass, 'r' for Radius
        Y_char: String alphabet (character) to depict Y quantity
            Eg 'm' for Mass, 'r' for Radius
        Y_min: Lower bound for Y. Default=None.
                  If None, uses: np.log10(max(min(Y - Y_sigma), 0.01))
        Y_max: Upper bound for Y. Default=None.
                  If None, uses: np.log10(max(Y + Y_sigma))
        X_min: Lower bound for X. Default=None.
                  If None, uses: max(np.log10(min(X - X_sigma)), -0.3)
        X_max: Upper bound for X. Default=None.
                  If None, uses: np.log10(max(X + X_sigma))
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

    LabelDictionary = {'X_label':X_label, 'Y_label':Y_label, 'X_char': X_char, 'Y_char':Y_char}
    with open(os.path.join(aux_output_location, 'AxesLabels.txt'), 'w') as f:
        print(LabelDictionary, file=f)

    message = """
	___  _________ _______   _______
	|  \/  || ___ \  ___\ \ / /  _  |
	| .  . || |_/ / |__  \ V /| | | |
	| |\/| ||    /|  __| /   \| | | |
	| |  | || |\ \| |___/ /^\ \ \_/ /
	\_|  |_/\_| \_\____/\/   \/\___/

    """
    _ = _logging(message=message, filepath=aux_output_location, verbose=verbose, append=True)

    message = 'Started for {} degrees at {}, using {} core/s'.format(select_deg, starttime, cores)
    _ = _logging(message=message, filepath=aux_output_location, verbose=verbose, append=True)

    n = len(Y)

    if len(Y) != len(X):
        print('Length of Y and X vectors must be the same')
    if len(Y) != len(Y_sigma) and (Y_sigma is not None):
        print('Length of Y and Y sigma vectors must be the same')
    if len(X) != len(X_sigma) and (X_sigma is not None):
        print('Length of X and X sigma vectors must be the same')

    if Y_min is None:
        if np.any(np.isnan(Y_sigma)):
            print('Provide {} Bounds'.format(Y_label))
        Y_min = np.log10(max(min(Y - Y_sigma), 0.01))
    if Y_max is None:
        if np.any(np.isnan(Y_sigma)):
            print('Provide {} Bounds'.format(Y_label))
        Y_max = np.log10(max(Y + Y_sigma))
    if X_min is None:
        if np.any(np.isnan(X_sigma)):
            print('Provide {} Bounds'.format(X_label))
        X_min = np.log10(min(np.abs(X - X_sigma)))
    if X_max is None:
        if np.any(np.isnan(X_sigma)):
            print('Provide {} Bounds'.format(X_label))
        X_max = np.log10(max(X + X_sigma))

    Y_sigma[(Y_sigma!=np.nan) & (Y_sigma[Y_sigma!=np.nan] < YSigmaLimit)] = np.nan
    X_sigma[(X_sigma!=np.nan) & (X_sigma[X_sigma!=np.nan] < XSigmaLimit)] = np.nan

    if degree_max == None:
        degree_max = int(n/np.log10(n)) + 2
    else:
        degree_max = int(degree_max)

    Y_bounds = np.array([Y_min, Y_max])
    X_bounds = np.array([X_min, X_max])


    t = Table([Y, Y_sigma, X, X_sigma], names=(Y_char, Y_char+'_sigma', X_char, X_char+'_sigma'))
    t.write(os.path.join(input_location, 'XY_inputs.csv'), overwrite=True)
    np.savetxt(os.path.join(input_location, 'Y_bounds.txt'),Y_bounds, comments='#', header='Minimum and maximum {} (log10)'.format(Y_label))
    np.savetxt(os.path.join(input_location, 'X_bounds.txt'),X_bounds, comments='#', header='Minimum and maximum {} (log10)'.format(X_label))

    ###########################################################
    ## Step 1: Select number of degrees based on cross validation (CV), AIC or BIC methods.

    if select_deg == 'cv':
        # Use the CV method with training and test dataset to maximize log likelihood.
        if k_fold == None:
            if n//10 > 5:
                k_fold = 10
            else:
                k_fold = 5

        message = 'Picked {} k-folds'.format(k_fold)
        _ = _logging(message=message, filepath=aux_output_location, verbose=verbose, append=True)

        deg_choose = run_cross_validation(Y=Y, X=X, Y_sigma=Y_sigma, X_sigma=X_sigma,
                                        X_char=X_char, Y_char=Y_char,
                                        Y_bounds=Y_bounds, X_bounds=X_bounds,
                                        degree_max=degree_max, k_fold=k_fold, cores=cores, save_path=aux_output_location, abs_tol=abs_tol, verbose=verbose)

        message = 'Finished CV. Picked {} degrees by maximizing likelihood\n'.format(deg_choose)
        _ = _logging(message=message, filepath=aux_output_location, verbose=verbose, append=True)

    elif select_deg == 'aic' :
        # Minimize the AIC
        degree_candidates = np.linspace(5, degree_max, 10, dtype = int)
        aic = np.array([MLE_fit(Y=Y, X=X, Y_sigma=Y_sigma, X_sigma=X_sigma,
                        X_char=X_char, Y_char=Y_char,
                        Y_bounds=Y_bounds, X_bounds=X_bounds, deg=d, abs_tol=abs_tol,
                        save_path=aux_output_location, verbose=verbose)['aic'] for d in degree_candidates])

        deg_choose = degree_candidates[np.argmin(aic)]

        message = 'Finished AIC check. Picked {} degrees by minimizing AIC '.format(deg_choose)
        _ = _logging(message=message, filepath=aux_output_location, verbose=verbose, append=True)

        np.savetxt(os.path.join(aux_output_location,'AIC_degreechoose.txt'),np.array([degree_candidates,aic]))


    elif select_deg == 'bic':
        # Minimize the BIC
        degree_candidates = np.linspace(5, degree_max, 10, dtype = int)
        bic = np.array([MLE_fit(Y=Y, X=X, Y_sigma=Y_sigma, X_sigma=X_sigma,
                        X_char=X_char, Y_char=Y_char,
                        Y_bounds=Y_bounds, X_bounds=X_bounds, deg=d,
                        abs_tol=abs_tol, save_path=aux_output_location, verbose=verbose)['bic'] for d in degree_candidates])

        deg_choose = degree_candidates[np.argmin(bic)]

        message = 'Finished BIC check. Picked {} degrees by minimizing BIC '.format(deg_choose)
        _ = _logging(message=message, filepath=aux_output_location, verbose=verbose, append=True)

        np.savetxt(os.path.join(aux_output_location,'BIC_degreechoose.txt'),np.array([degree_candidates,bic]))


    elif isinstance(select_deg, (int,float)):
        # Use user defined value
        deg_choose = select_deg

    else :
        print('Error: Incorrect input for select_deg. Please read docstring for valid inputs.')
        return 0

    ###########################################################
    ## Step 2: Estimate the full model without bootstrap

    print('Running full dataset MLE before bootstrap')


    message = 'Running full dataset MLE before bootstrap\n'
    _ = _logging(message=message, filepath=aux_output_location, verbose=verbose, append=True)

    initialfit_result = MLE_fit(Y=Y, X=X, Y_sigma=Y_sigma, X_sigma=X_sigma,
                            Y_bounds=Y_bounds, X_bounds=X_bounds,
                            X_char=X_char, Y_char=Y_char,
                            deg=int(deg_choose), abs_tol=abs_tol, save_path=aux_output_location,
                            calc_joint_dist = True, verbose=verbose)

    message = 'Finished full dataset MLE run at {}\n'.format(datetime.datetime.now())
    _ = _logging(message=message, filepath=aux_output_location, verbose=verbose, append=True)

    _save_dictionary(dictionary=initialfit_result, output_location=output_location, bootstrap=False,
                    X_char=X_char, Y_char=Y_char, X_label=X_label, Y_label=Y_label)

    ###########################################################
    ## Step 3: Run Bootstrap
    if num_boot == 0:
        message='Bootstrap not run since num_boot = 0'
        _ = _logging(message=message, filepath=aux_output_location, verbose=verbose, append=True)
        return initialfit_result
    else:
        # Generate iterator for using multiprocessing Pool.imap
        n_boot_iter = (np.random.choice(n, n, replace=True) for i in range(num_boot))
        inputs = ((Y[n_boot], X[n_boot], Y_sigma[n_boot], X_sigma[n_boot], Y_char, X_char,
                Y_bounds, X_bounds, deg_choose, abs_tol, aux_output_location, verbose) for n_boot in n_boot_iter)

        message = '\n\n==============\nRunning {} bootstraps for the MLE code with degree = {}, using {} thread/s.\n==============\n\n'.format(str(num_boot),
                    str(deg_choose),str(cores))
        _ = _logging(message=message, filepath=aux_output_location, verbose=verbose, append=True)



        # Parallelize the bootstraps
        pool = Pool(processes=cores)
        bootstrap_results = list(pool.imap(_bootsample_mle,inputs))

        _save_dictionary(dictionary=bootstrap_results, output_location=output_location, bootstrap=True,
                            X_char=X_char, Y_char=Y_char, X_label=X_label, Y_label=Y_label)


        message = 'Finished bootstrap at {}'.format(datetime.datetime.now())
        _ = _logging(message=message, filepath=aux_output_location, verbose=verbose, append=True)


        endtime = datetime.datetime.now()
        print(endtime - starttime)


        message = 'Ended run at {}'.format(endtime)
        _ = _logging(message=message, filepath=aux_output_location, verbose=verbose, append=True)


        message = """
	 _______ _    _ ______   ______ _   _ _____
	 |__   __| |  | |  ____| |  ____| \ | |  __ |
	    | |  | |__| | |__    | |__  |  \| | |  | |
	    | |  |  __  |  __|   |  __| | . ` | |  | |
	    | |  | |  | | |____  | |____| |\  | |__| |
	    |_|  |_|  |_|______| |______|_| \_|_____/
        """
        _ = _logging(message=message, filepath=aux_output_location, verbose=verbose, append=True)




        return initialfit_result, bootstrap_results


def _bootsample_mle(inputs):
    """
    To bootstrap the data and run MLE. Serves as input to the parallelizing function.
    \nINPUTS:
        inputs : Variable required for mapping for parallel processing.
        inputs is a tuple with the following components :
                    Y: Numpy array of Y measurements. In LINEAR SCALE.
                    X: Numpy array of X measurements. In LINEAR SCALE.
                    Y_sigma: Numpy array of Y uncertainties. Assumes symmetrical uncertainty. In LINEAR SCALE.
                    X_sigma: Numpy array of X uncertainties. Assumes symmetrical uncertainty. In LINEAR SCALE.
                    X_char: String alphabet (character) to depict X quantity.
                        Eg 'm' for Mass, 'r' for Radius
                    Y_char: String alphabet (character) to depict Y quantity
                        Eg 'm' for Mass, 'r' for Radius
                    Y_bounds: Bounds for the Y.
                    X_bounds: Bounds for the X.
                    deg: Degree chosen for the beta densities.
                    abs_tol: Absolute tolerance to be used for the numerical integration for product of normal and beta distribution.
                             Default : 1e-8
                    save_path: Folder name (+path) to save results in. Eg. save_path='~/mrexo_working/trial_result'
                    verbose: Keyword specifying verbosity
    OUTPUTS:

        XY_boot :Output dictionary from bootstrap run using Maximum Likelihood Estimation. Its keys are  -
                'weights' : Weights for Beta densities from bootstrap run.
                'aic' : Akaike Information Criterion from bootstrap run.
                'bic' : Bayesian Information Criterion from bootstrap run.
                'Y_points' : Sequence of Y points for initial fitting w/o bootstrap.
                'X_points' : Sequence of X points for initial fitting w/o bootstrap.
                'Y_cond_X' : Conditional distribution of Y given X from bootstrap run.
                'Y_cond_X_var' : Variance for the Conditional distribution of Y given X from bootstrap run.
                'Y_cond_X_quantile' : Quantiles for the Conditional distribution of Y given X from bootstrap run.
                'X_cond_Y' : Conditional distribution of X given Y from bootstrap run.
                'X_cond_Y_var' : Variance for the Conditional distribution of X given Y from bootstrap run.
                'X_cond_Y_quantile' : Quantiles for the Conditional distribution of X given Y from bootstrap run.
                'X_marg' : Marginalized X distribution from bootstrap run.
                'Y_marg' : Marginalized Y distribution from bootstrap run.
    """

    XY_boot = MLE_fit(Y=inputs[0], X=inputs[1],
                    Y_sigma=inputs[2], X_sigma=inputs[3],
                    Y_char=inputs[4], X_char=inputs[5],
                    Y_bounds=inputs[6], X_bounds=inputs[7],
                    deg=inputs[8],
                    abs_tol=inputs[9], save_path=inputs[10], verbose=inputs[11])

    return XY_boot

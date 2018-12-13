#%cd "C:/Users/shbhu/Documents/Git/Py_mass_radius_working/PyCode"
import numpy as np
from multiprocessing import Pool,cpu_count
import os
from astropy.table import Table
import datetime
#from shutil import copyfile

from .mle_utils import MLE_fit
from .cross_validate import run_cross_validation
from .utils import save_dictionary


def fit_mr_relation(Mass, Mass_sigma, Radius, Radius_sigma, save_path,
                    Mass_min=None, Mass_max=None, Radius_min=None, Radius_max=None,
                    select_deg=55, degree_max=None, k_fold=None, num_boot=100,
                    cores=1, abs_tol=1e-8):
    '''
    Fit a mass and radius relationship using a non parametric approach with beta densities

    INPUTS:
        Mass: Numpy array of mass measurements. In LINEAR SCALE.
        Mass_sigma: Numpy array of mass uncertainties. Assumes symmetrical uncertainty. In LINEAR SCALE.
        Radius: Numpy array of radius measurements. In LINEAR SCALE.
        Radius_sigma: Numpy array of radius uncertainties. Assumes symmetrical uncertainty. In LINEAR SCALE.
        save_path: Folder name (+path) to save results in.
                   Eg. save_path = '~/mrexo_working/trial_result' will create the 'trial_result' results folder in mrexo_working
        Mass_min: Lower bound for mass. Default=None. If None, uses: np.log10(max(min(Mass - Mass_sigma), 0.01))
        Mass_max: Upper bound for mass. Default=None. If None, uses: np.log10(max(Mass + Mass_sigma))
        Radius_min: Lower bound for radius. Default=None. If None, uses: max(np.log10(min(Radius - Radius_sigma)), -0.3)
        Radius_max: Upper bound for radius. Default=None. If None, uses: np.log10(max(Radius + Radius_sigma))

        select_deg: The number of degrees for the beta densities.
                            if select_deg= "cv": Use cross validation to find the optimal number of  degrees.
                            if select_deg= "aic": Use AIC minimization to find the optimal number of degrees.
                            if select_deg= "bic": Use BIC minimization to find the optimal number of degrees.
                            if select_deg= INTEGER: Use that number and skip the
                                             optimization process to find the number of degrees.
                            NOTE: Use AIC or BIC optimization only for large (> 200) sample sizes.
        degree_max: Maximum degree used for cross-validation/AIC/BIC. Type: Integer.
                    Default=None. If None, uses: n/np.log10(n), where n is the number of data points.
        k_fold: If using cross validation method, use k_fold (integer) number of folds. Default=None.
                If None, uses: 10 folds for n > 60, 5 folds otherwise. Eg. k_fold=12
        num_boot: Number of bootstraps to perform. Default=100. num_boot must be greater than 1.
        cores: Number of cores for parallel processing. This is used in the
                bootstrap and the cross validation. Default=1.
                To use all the cores in the CPU, cores=cpu_count() (from multiprocessing import cpu_count)
        abs_tol : Absolute tolerance to be used for the numerical integration for product of normal and beta distribution.
                Default : 1e-8

    OUTPUTS:
        initialfit_result : Output dictionary from initial fitting without bootstrap using Maximum Likelihood Estimation.
                            The keys in the dictionary are -
                            'weights' : Weights for Beta densities from initial fitting w/o bootstrap.
                            'aic' : Akaike Information Criterion from initial fitting w/o bootstrap.
                            'bic' : Bayesian Information Criterion from initial fitting w/o bootstrap.
                            'M_points' : Sequence of mass points for initial fitting w/o bootstrap.
                            'R_points' : Sequence of radius points for initial fitting w/o bootstrap.
                            'M_cond_R' : Conditional distribution of mass given radius from initial fitting w/o bootstrap.
                            'M_cond_R_var' : Variance for the Conditional distribution of mass given radius from initial fitting w/o bootstrap.
                            'M_cond_R_quantile' : Quantiles for the Conditional distribution of mass given radius from initial fitting w/o bootstrap.
                            'R_cond_M' : Conditional distribution of radius given mass from initial fitting w/o bootstrap.
                            'R_cond_M_var' : Variance for the Conditional distribution of radius given mass from initial fitting w/o bootstrap.
                            'R_cond_M_quantile' : Quantiles for the Conditional distribution of radius given mass from initial fitting w/o bootstrap.
                            'joint_dist' : Joint distribution of mass AND radius.


        if num_boot > 2:
        bootstrap_results : Output dictionary from bootstrap run using Maximum Likelihood Estimation.
                            'weights' : Weights for Beta densities from bootstrap run.
                            'aic' : Akaike Information Criterion from bootstrap run.
                            'bic' : Bayesian Information Criterion from bootstrap run.
                            'M_points' : Sequence of mass points for initial fitting w/o bootstrap.
                            'R_points' : Sequence of radius points for initial fitting w/o bootstrap.
                            'M_cond_R' : Conditional distribution of mass given radius from bootstrap run.
                            'M_cond_R_var' : Variance for the Conditional distribution of mass given radius from bootstrap run.
                            'M_cond_R_quantile' : Quantiles for the Conditional distribution of mass given radius from bootstrap run.
                            'R_cond_M' : Conditional distribution of radius given mass from bootstrap run.
                            'R_cond_M_var' : Variance for the Conditional distribution of radius given mass from bootstrap run.
                            'R_cond_M_quantile' : Quantiles for the Conditional distribution of radius given mass from bootstrap run.


        EXAMPLE:
            # Example to fit a Mass Radius relationship with 2 CPU cores, using 12 degrees, and 50 bootstraps.
            import os
            from astropy.table import Table
            import numpy as np
            from mrexo import fit_mr_relation

            pwd = '~/mrexo_working/'

            t = Table.read(os.path.join(pwd,'Cool_stars_20181109.csv'))

            # Symmetrical errorbars
            Mass_sigma = (abs(t['pl_masseerr1']) + abs(t['pl_masseerr2']))/2
            Radius_sigma = (abs(t['pl_radeerr1']) + abs(t['pl_radeerr2']))/2

            # In Earth units
            Mass = np.array(t['pl_masse'])
            Radius = np.array(t['pl_rade'])

            # Directory to store results in
            result_dir = os.path.join(pwd,'Results_deg_12')

            initialfit_result, bootstrap_results = fit_mr_relation(Mass=Mass, Mass_sigma=Mass_sigma,
                                                    Radius=Radius, Radius_sigma=Radius_sigma,
                                                    save_path=result_dir, select_deg=12,
                                                    num_boot=50, cores=2)
    '''

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

    t = Table([Mass, Mass_sigma, Radius, Radius_sigma], names=('pl_masse', 'pl_masseerr1', 'pl_rade', 'pl_radeerr1'))
    t.write(os.path.join(input_location, 'MR_inputs.csv'), overwrite=True)

    with open(os.path.join(aux_output_location,'log_file.txt'),'a') as f:
       f.write('Started for {} degrees at {}, using {} core/s'.format(select_deg, starttime, cores))

    n = len(Mass)

    if len(Mass) != len(Radius):
        print('Length of Mass and Radius vectors must be the same')
    if len(Mass) != len(Mass_sigma) and (Mass_sigma is not None):
        print('Length of Mass and Mass sigma vectors must be the same')
    if len(Radius) != len(Radius_sigma) and (Radius_sigma is not None):
        print('Length of Radius and Radius sigma vectors must be the same')

    if Mass_min is None:
        Mass_min = np.log10(min(min(Mass - Mass_sigma), 0.01))
    if Mass_max is None:
        Mass_max = np.log10(max(Mass + Mass_sigma))
    if Radius_min is None:
        Radius_min = max(np.log10(min(Radius - Radius_sigma)), -0.3)
    if Radius_max is None:
        Radius_max = np.log10(max(Radius + Radius_sigma))

    if degree_max == None:
        degree_max = int(n/np.log10(n))

    Mass_bounds = np.array([Mass_min, Mass_max])
    Radius_bounds = np.array([Radius_min, Radius_max])

    if type(degree_max) != int:
        degree_max = int(degree_max)

    ###########################################################
    ## Step 1: Select number of degrees based on cross validation (CV), AIC or BIC methods.

    if select_deg == 'cv':
        # Use the CV method with training and test dataset to maximize log likelihood.
        if k_fold == None:
            if n//10 > 5:
                k_fold = 10
            else:
                k_fold = 5
            print('Picked {} k-folds'.format(k_fold))

        deg_choose = run_cross_validation(Mass=Mass, Radius=Radius, Mass_sigma=Mass_sigma, Radius_sigma=Radius_sigma,
                                        Mass_bounds=Mass_bounds, Radius_bounds=Radius_bounds,
                                        degree_max=degree_max, k_fold=k_fold, cores=cores, save_path=aux_output_location, abs_tol=abs_tol)

        with open(os.path.join(aux_output_location,'log_file.txt'),'a') as f:
            f.write('Finished CV. Picked {} degrees by maximizing likelihood'.format({deg_choose}))

    elif select_deg == 'aic' :
        # Minimize the AIC
        degree_candidates = np.linspace(5, degree_max, 12, dtype = int)
        aic = np.array([MLE_fit(Mass=Mass, Radius=Radius, Mass_sigma=Mass_sigma, Radius_sigma=Radius_sigma,
                        Mass_bounds=Mass_bounds, Radius_bounds=Radius_bounds, deg=d, abs_tol=abs_tol, save_path=aux_output_location)['aic'] for d in degree_candidates])

        deg_choose = degree_candidates[np.argmin(aic)]

        with open(os.path.join(aux_output_location,'log_file.txt'),'a') as f:
            f.write('Finished AIC check. Picked {} degrees by minimizing AIC'.format({deg_choose}))

        print('Finished AIC check. Picked {} degrees by minimizing AIC'.format({deg_choose}))
        np.savetxt(os.path.join(aux_output_location,'AIC_degreechoose.txt'),np.array([degree_candidates,aic]))


    elif select_deg == 'bic':
        # Minimize the BIC
        degree_candidates = np.linspace(5, degree_max, 12, dtype = int)
        bic = np.array([MLE_fit(Mass=Mass, Radius=Radius, Mass_sigma=Mass_sigma, Radius_sigma=Radius_sigma,
                        Mass_bounds=Mass_bounds, Radius_bounds=Radius_bounds, deg=d, abs_tol=abs_tol, save_path=aux_output_location)['bic'] for d in degree_candidates])

        deg_choose = degree_candidates[np.argmin(bic)]

        with open(os.path.join(aux_output_location,'log_file.txt'),'a') as f:
            f.write('Finished BIC check. Picked {} degrees by minimizing BIC'.format({deg_choose}))

        print('Finished BIC check. Picked {} degrees by minimizing BIC'.format({deg_choose}))
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
    initialfit_result = MLE_fit(Mass=Mass, Radius=Radius, Mass_sigma=Mass_sigma, Radius_sigma=Radius_sigma,
                            Mass_bounds=Mass_bounds, Radius_bounds=Radius_bounds,  deg=int(deg_choose), abs_tol=abs_tol, save_path=aux_output_location,
                            calc_joint_dist = True)

    with open(os.path.join(aux_output_location,'log_file.txt'),'a') as f:
       f.write('Finished full dataset MLE run at {}\n'.format(datetime.datetime.now()))

    np.savetxt(os.path.join(input_location, 'Mass_bounds.txt'),Mass_bounds, comments='#', header='Minimum mass and maximum mass (log10)')
    np.savetxt(os.path.join(input_location, 'Radius_bounds.txt'),Radius_bounds, comments='#', header='Minimum radius and maximum radius (log10)')

    save_dictionary(dictionary=initialfit_result, output_location=output_location, bootstrap=False)

    ###########################################################
    ## Step 3: Run Bootstrap
    if num_boot == 0:
        print('Bootstrap not run since num_boot = 0')
        return initialfit_result
    else:
        # Generate iterator for using multiprocessing Pool.imap
        n_boot_iter = (np.random.choice(n, n, replace=True) for i in range(num_boot))
        inputs = ((Mass[n_boot], Radius[n_boot], Mass_sigma[n_boot], Radius_sigma[n_boot],
                Mass_bounds, Radius_bounds, deg_choose, abs_tol, aux_output_location) for n_boot in n_boot_iter)

        print('Running {} bootstraps for the MLE code with degree = {}, using {} thread/s.'.format(str(num_boot),str(deg_choose),str(cores)))

        with open(os.path.join(aux_output_location,'log_file.txt'),'a') as f:
            f.write('Running {} bootstraps for the MLE code with degree = {}, using {} thread/s.'.format(str(num_boot),str(deg_choose),str(cores)))

        # Parallelize the bootstraps
        pool = Pool(processes=cores)
        bootstrap_results = list(pool.imap(bootsample_mle,inputs))

        save_dictionary(dictionary=bootstrap_results, output_location=output_location, bootstrap=True)

        print('Finished bootstrap at {}'.format(datetime.datetime.now()))

        endtime = datetime.datetime.now()
        print(endtime - starttime)

        with open(os.path.join(aux_output_location,'log_file.txt'),'a') as f:
            f.write('Ended run at {}\n'.format(endtime))


        return initialfit_result, bootstrap_results


def bootsample_mle(inputs):
    '''
    To bootstrap the data and run MLE. Serves as input to the parallelizing function.
    INPUTS:
        inputs : Variable required for mapping for parallel processing.
        inputs is a tuple with the following components :
                    Mass: Numpy array of mass measurements. In LINEAR SCALE.
                    Radius: Numpy array of radius measurements. In LINEAR SCALE.
                    Mass_sigma: Numpy array of mass uncertainties. Assumes symmetrical uncertainty. In LINEAR SCALE.
                    Radius_sigma: Numpy array of radius uncertainties. Assumes symmetrical uncertainty. In LINEAR SCALE.
                    Mass_bounds: Bounds for the mass.
                    Radius_bounds: Bounds for the radius.
                    deg: Degree chosen for the beta densities.
                    abs_tol: Absolute tolerance to be used for the numerical integration for product of normal and beta distribution.
                             Default : 1e-8
                    save_path: Folder name (+path) to save results in. Eg. save_path='~/mrexo_working/trial_result'
    OUTPUTS:
        MR_boot :Output dictionary from bootstrap run using Maximum Likelihood Estimation. Its keys are  -
                'weights' : Weights for Beta densities from bootstrap run.
                'aic' : Akaike Information Criterion from bootstrap run.
                'bic' : Bayesian Information Criterion from bootstrap run.
                'M_points' : Sequence of mass points for initial fitting w/o bootstrap.
                'R_points' : Sequence of radius points for initial fitting w/o bootstrap.
                'M_cond_R' : Conditional distribution of mass given radius from bootstrap run.
                'M_cond_R_var' : Variance for the Conditional distribution of mass given radius from bootstrap run.
                'M_cond_R_quantile' : Quantiles for the Conditional distribution of mass given radius from bootstrap run.
                'R_cond_M' : Conditional distribution of radius given mass from bootstrap run.
                'R_cond_M_var' : Variance for the Conditional distribution of radius given mass from bootstrap run.
                'R_cond_M_quantile' : Quantiles for the Conditional distribution of radius given mass from bootstrap run.
                'Radius_marg' : Marginalized radius distribution from bootstrap run.
                'Mass_marg' : Marginalized mass distribution from bootstrap run.
    '''

    MR_boot = MLE_fit(Mass=inputs[0], Radius=inputs[1],
                    Mass_sigma=inputs[2], Radius_sigma=inputs[3],
                    Mass_bounds=inputs[4], Radius_bounds=inputs[5],
                    deg=inputs[6], abs_tol=inputs[7], save_path=inputs[8])

    return MR_boot

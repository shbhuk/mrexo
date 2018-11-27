#%cd "C:/Users/shbhu/Documents/Git/Py_mass_radius_working/PyCode"
import numpy as np
from multiprocessing import Pool,cpu_count
import os

import datetime
from shutil import copyfile

from .mle_utils import MLE_fit
from .cross_validate import run_cross_validation



def fit_mr_relation(Mass, Mass_sigma, Radius, Radius_sigma, Mass_max = None, Mass_min = None, Radius_max = None, Radius_min = None,
                    degree_max = 60, select_deg = 55, Log = False, k_fold = None, num_boot = 100,
                    cores = cpu_count(), location = os.path.dirname(__file__), abs_tol = 1e-10):
    '''
    Predict the Mass and Radius relationship
    INPUT:

        Mass: Mass measurements
        Radius : Radius measurements
        Mass_sigma: measurement errors for the data, if no measuremnet error,
                    it is NULL
        Radius_sigma
        Mass_max: the upper bound for mass. Default = None
        Mass_min: the lower bound for mass. Default = None
        Radius_max: the upper bound for radius. Default = None
        Radius_min: the upper bound for radius. Default = None
        degree_max: INTEGER the maximum degree used for cross-validation/AIC/BIC. Default = 60. Suggested value = n/log(n)
        select_deg: if input "cv": cross validation
                            if input "aic": aic method
                            if input "bic": bic method
                            if input a number: default using that number and
                            skip the select process
        Log: is the data transformed into a log scale if Log = True. Default = False
        k_fold: number of fold used for cross validation. Default = 10
        num_boot: number of bootstrap replication. Default = 100
        cores: this program uses parallel computing for bootstrap. Default = 1
        location : The location for the log file
        abs_tol : Defined for integration in MLE_fit()

    '''


    starttime = datetime.datetime.now()
    print('Started for {} degrees at {}, using {} core/s'.format(select_deg, starttime, cores))

    if not os.path.exists(location):
        os.mkdir(location)

    with open(os.path.join(location,'log_file.txt'),'a') as f:
       f.write('Started for {} degrees at {}, using {} core/s'.format(select_deg, starttime, cores))


    copyfile(os.path.join(os.path.dirname(__file__),os.path.basename(__file__)), os.path.join(location,os.path.basename(__file__)))
    #copyfile(os.path.join(os.path.dirname(location),'mle_utils.py os.path.join(location,'mle_utils.py'))

    n = len(Mass)

    if len(Mass) != len(Radius):
        print('Length of Mass and Radius vectors must be the same')
    if len(Mass) != len(Mass_sigma) and (Mass_sigma is not None):
        print('Length of Mass and Mass sigma vectors must be the same')
    if len(Radius) != len(Radius_sigma) and (Radius_sigma is not None):
        print('Length of Radius and Radius sigma vectors must be the same')

    if Mass_min is None:
        Mass_min = np.min(Mass) - np.max(Mass_sigma)/np.sqrt(n)
    if Mass_max is None:
        Mass_max = np.max(Mass) + np.max(Mass_sigma)/np.sqrt(n)
    if Radius_min is None:
        Radius_min = np.min(Radius) - np.max(Radius_sigma)/np.sqrt(n)
    if Radius_max is None:
        Radius_max = np.max(Radius) + np.max(Radius_sigma)/np.sqrt(n)

    Mass_bounds = np.array([Mass_max,Mass_min])
    Radius_bounds = np.array([Radius_max,Radius_min])

    if type(degree_max) != int:
        degree_max = int(degree_max)

    ###########################################################
    ## Step 1: Select number of degree based on cross validation, aic or bic methods.

    if select_deg == 'cv':

        if k_fold == None:
            if n//10 > 5:
                k_fold = 10
            else:
                k_fold = 5
            print('Picked {} k-folds'.format(k_fold))


        deg_choose = run_cross_validation(Mass = Mass, Radius = Radius, Mass_sigma = Mass_sigma, Radius_sigma = Radius_sigma,
                                        Mass_bounds = Mass_bounds, Radius_bounds = Radius_bounds, Log = Log,
                                        degree_max = degree_max, k_fold = k_fold, cores = cores, location = location, abs_tol = abs_tol)



        with open(os.path.join(location,'log_file.txt'),'a') as f:
            f.write('Finished CV. Picked {} degrees by maximizing likelihood'.format({deg_choose}))

    elif select_deg == 'aic' :
        degree_candidates = np.linspace(5, degree_max, 12, dtype = int)
        aic = np.array([MLE_fit(Mass = Mass, Radius = Radius, Mass_sigma = Mass_sigma, Radius_sigma = Radius_sigma,
                        Mass_bounds = Mass_bounds, Radius_bounds = Radius_bounds, Log = Log, deg = d, abs_tol = abs_tol, location = location)['aic'] for d in degree_candidates])

        deg_choose = degree_candidates[np.argmin(aic)]

        with open(os.path.join(location,'log_file.txt'),'a') as f:
            f.write('Finished AIC check. Picked {} degrees by minimizing AIC'.format({deg_choose}))

        print('Finished AIC check. Picked {} degrees by minimizing AIC'.format({deg_choose}))
        np.savetxt(os.path.join(location,'AIC_degreechoose.txt'),np.array([degree_candidates,aic]))


    elif select_deg == 'bic':
        degree_candidates = np.linspace(5, degree_max, 12, dtype = int)
        bic = np.array([MLE_fit(Mass = Mass, Radius = Radius, Mass_sigma = Mass_sigma, Radius_sigma = Radius_sigma,
                        Mass_bounds = Mass_bounds, Radius_bounds = Radius_bounds, Log = Log, deg = d, abs_tol = abs_tol, location = location)['bic'] for d in degree_candidates])

        deg_choose = degree_candidates[np.argmin(bic)]

        with open(os.path.join(location,'log_file.txt'),'a') as f:
            f.write('Finished BIC check. Picked {} degrees by minimizing BIC'.format({deg_choose}))

        print('Finished BIC check. Picked {} degrees by minimizing BIC'.format({deg_choose}))
        np.savetxt(os.path.join(location,'BIC_degreechoose.txt'),np.array([degree_candidates,bic]))


    elif isinstance(select_deg, (int,float)):
        deg_choose = select_deg

    else :
        print('Error: Incorrect input for select_deg')
        return 0

    ###########################################################
    ## Step 2: Estimate the model

    print('Running full dataset MLE before bootstrap')
    fullMLEresult = MLE_fit(Mass = Mass, Radius = Radius, Mass_sigma = Mass_sigma, Radius_sigma = Radius_sigma,
                            Mass_bounds = Mass_bounds, Radius_bounds = Radius_bounds, Log = Log, deg = int(deg_choose), abs_tol = abs_tol, location = location)


    with open(os.path.join(location,'log_file.txt'),'a') as f:
       f.write('Finished full dataset MLE run at {}\n'.format(datetime.datetime.now()))


    weights = fullMLEresult['weights']
    aic = fullMLEresult['aic']
    bic = fullMLEresult['bic']
    M_points =  fullMLEresult['M_points']
    R_points = fullMLEresult['R_points']
    M_cond_R = fullMLEresult['M_cond_R']
    M_cond_R_var = fullMLEresult['M_cond_R_var']
    M_cond_R_lower = fullMLEresult['M_cond_R_quantile'][:,0]
    M_cond_R_upper = fullMLEresult['M_cond_R_quantile'][:,1]
    R_cond_M = fullMLEresult['R_cond_M']
    R_cond_M_var = fullMLEresult['R_cond_M_var']
    R_cond_M_lower = fullMLEresult['R_cond_M_quantile'][:,0]
    R_cond_M_upper = fullMLEresult['R_cond_M_quantile'][:,1]
    Radius_marg = fullMLEresult['Radius_marg']
    Mass_marg = fullMLEresult['Mass_marg']

    np.savetxt(os.path.join(location,'weights.txt'),weights)
    np.savetxt(os.path.join(location,'aic.txt'),[aic])
    np.savetxt(os.path.join(location,'bic.txt'),[bic])
    np.savetxt(os.path.join(location,'M_points.txt'),M_points)
    np.savetxt(os.path.join(location,'R_points.txt'),R_points)
    np.savetxt(os.path.join(location,'M_cond_R.txt'),M_cond_R)
    np.savetxt(os.path.join(location,'M_cond_R_var.txt'),M_cond_R_var)
    np.savetxt(os.path.join(location,'M_cond_R_lower.txt'),M_cond_R_lower)
    np.savetxt(os.path.join(location,'M_cond_R_upper.txt'),M_cond_R_upper)
    np.savetxt(os.path.join(location,'R_cond_M.txt'),R_cond_M)
    np.savetxt(os.path.join(location,'R_cond_M_var.txt'),R_cond_M_var)
    np.savetxt(os.path.join(location,'R_cond_M_lower.txt'),R_cond_M_lower)
    np.savetxt(os.path.join(location,'R_cond_M_upper.txt'),R_cond_M_upper)
    np.savetxt(os.path.join(location,'Radius_marg.txt'),Radius_marg)
    np.savetxt(os.path.join(location,'Mass_marg.txt'),Mass_marg)

    n_boot_iter = (np.random.choice(n, n, replace = True) for i in range(num_boot))
    inputs = ((Mass[n_boot], Radius[n_boot], Mass_sigma[n_boot], Radius_sigma[n_boot],
            Mass_bounds, Radius_bounds, Log, deg_choose, abs_tol, location) for n_boot in n_boot_iter)

    print('Running {} bootstraps for the MLE code with degree = {}, using {} thread/s.'.format(str(num_boot),str(deg_choose),str(cores)))

    with open(os.path.join(location,'log_file.txt'),'a') as f:
       f.write('Running {} bootstraps for the MLE code with degree = {}, using {} thread/s.'.format(str(num_boot),str(deg_choose),str(cores)))

    pool = Pool(processes = cores)
    results = list(pool.imap(bootsample_mle,inputs))

    print('Finished bootstrap at {}'.format(datetime.datetime.now()))

    weights_boot = np.array([x['weights'] for x in results])
    aic_boot = np.array([x['aic'] for x in results])
    bic_boot = np.array([x['bic'] for x in results])
    M_points_boot =  np.array([x['M_points'] for x in results])
    R_points_boot = np.array([x['R_points'] for x in results])
    M_cond_R_boot = np.array([x['M_cond_R'] for x in results])
    M_cond_R_var_boot = np.array([x['M_cond_R_var'] for x in results])
    M_cond_R_lower_boot = np.array([x['M_cond_R_quantile'][:,0] for x in results])
    M_cond_R_upper_boot = np.array([x['M_cond_R_quantile'][:,1] for x in results])
    R_cond_M_boot = np.array([x['R_cond_M'] for x in results])
    R_cond_M_var_boot = np.array([x['R_cond_M_var'] for x in results])
    R_cond_M_lower_boot = np.array([x['R_cond_M_quantile'][:,0] for x in results])
    R_cond_M_upper_boot = np.array([x['R_cond_M_quantile'][:,1] for x in results])
    Radius_marg_boot = np.array([x['Radius_marg'] for x in results])
    Mass_marg_boot = np.array([x['Mass_marg'] for x in results])

    np.savetxt(os.path.join(location,'weights_boot.txt'),weights_boot)
    np.savetxt(os.path.join(location,'aic_boot.txt'),aic_boot)
    np.savetxt(os.path.join(location,'bic_boot.txt'),bic_boot)
    np.savetxt(os.path.join(location,'M_points_boot.txt'),M_points_boot)
    np.savetxt(os.path.join(location,'R_points_boot.txt'),R_points_boot)
    np.savetxt(os.path.join(location,'M_cond_R_boot.txt'),M_cond_R_boot)
    np.savetxt(os.path.join(location,'M_cond_R_var_boot.txt'),M_cond_R_var_boot)
    np.savetxt(os.path.join(location,'M_cond_R_lower_boot.txt'),M_cond_R_lower_boot)
    np.savetxt(os.path.join(location,'M_cond_R_upper_boot.txt'),M_cond_R_upper_boot)
    np.savetxt(os.path.join(location,'R_cond_M_boot.txt'),R_cond_M_boot)
    np.savetxt(os.path.join(location,'R_cond_M_var_boot.txt'),R_cond_M_var_boot)
    np.savetxt(os.path.join(location,'R_cond_M_lower_boot.txt'),R_cond_M_lower_boot)
    np.savetxt(os.path.join(location,'R_cond_M_upper_boot.txt'),R_cond_M_upper_boot)
    np.savetxt(os.path.join(location,'Radius_marg_boot.txt'),Radius_marg_boot)
    np.savetxt(os.path.join(location,'Mass_marg_boot.txt'),Mass_marg_boot)

    endtime = datetime.datetime.now()
    print(endtime - starttime)

    with open(os.path.join(location,'log_file.txt'),'a') as f:
       f.write('Ended run at {}\n'.format(endtime))


    return results




def bootsample_mle(inputs):
    '''
    To bootstrap the data and run MLE
    Input:
        inputs : Variable required for mapping for parallel processing
    '''

    MR_boot = MLE_fit(Mass = inputs[0], Radius = inputs[1],
                    Mass_sigma = inputs[2], Radius_sigma = inputs[3],
                    Mass_bounds = inputs[4], Radius_bounds = inputs[5],
                    Log = inputs[6], deg = inputs[7], abs_tol = inputs[8], location = inputs[9])

    return MR_boot

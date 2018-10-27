#%cd "C:/Users/shbhu/Documents/Git/Py_mass_radius_working/PyCode"
import numpy as np
from scipy.stats import beta,norm
from scipy.integrate import quad
from scipy.optimize import brentq as root
from astropy.table import Table
from scipy.optimize import minimize, fmin_slsqp
from multiprocessing import Pool,cpu_count
import os
import sys
import datetime
from shutil import copyfile

sys.path.append(os.path.dirname(__file__))
import importlib
from MLE_fit import MLE_fit
from Cross_Validation import cross_validation


t = Table.read(os.path.join(os.path.dirname(__file__),'MR_Kepler_170605_noanalytTTV_noupplim.csv'))
t = t.filled()

M_sigma = (abs(t['pl_masseerr1']) + abs(t['pl_masseerr2']))/2
R_sigma = (abs(t['pl_radeerr1']) + abs(t['pl_radeerr2']))/2

M_obs = np.array(t['pl_masse'])
R_obs = np.array(t['pl_rade'])

# bounds for Mass and Radius
Radius_min = -0.3
Radius_max = np.log10(max(R_obs) + np.std(R_obs)/np.sqrt(len(R_obs)))
Mass_min = np.log10(max(min(M_obs) - np.std(M_obs)/np.sqrt(len(M_obs)), 0.1))
Mass_max = np.log10(max(M_obs) + np.std(M_obs)/np.sqrt(len(M_obs)))
num_boot = 100
'''
Log = True
'''


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

def cv_parallel_fn(cv_input):
    i_fold, test_degree, indices_folded, n, rand_gen, Mass, Radius, Radius_sigma, Mass_sigma, Log, abs_tol, location, Mass_bounds, Radius_bounds = cv_input
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
    
    with open(os.path.join(location,'log_file.txt'),'a') as f:
       f.write('Running cross validation for {} degree check and {} th-fold'.format(test_degree, i_fold))
    
    like_pred = cross_validation(train_Radius = train_Radius, train_Mass = train_Mass, test_Radius = test_Radius, test_Mass = test_Mass,
                Mass_bounds = Mass_bounds, Radius_bounds = Radius_bounds, deg = test_degree,
                train_Radius_sigma = train_Radius_sigma, train_Mass_sigma = train_Mass_sigma,
                test_Radius_sigma = test_Radius_sigma, test_Mass_sigma = test_Mass_sigma,
                Log = Log, abs_tol = abs_tol, location = location)
    return like_pred


def run_cross_validation(Mass, Radius, Mass_sigma, Radius_sigma, Mass_bounds, Radius_bounds, Log = False, 
                        degree_max = 60, k_fold = 10, degree_candidate = None, 
                        cores = 1, location = os.path.dirname(__file__), abs_tol = 1e-10):
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

    Log: is the data transformed into a log scale if Log = True. Default = False
    degree_max: the maximum degree used for cross-validation/AIC/BIC. Default = 60. Suggested value = n/log10(n)
    k_fold: number of fold used for cross validation. Default = 10
    degree_candidate : Integer vector containing degrees to run cross validation check for. Default is None. 
                    If None, defaults to 
    cores: this program uses parallel computing for bootstrap. Default = 1
    location : The location for the log file
    abs_tol : Defined for integration in MLE_fit()
    
    '''
    if degree_candidate == None:
        degree_candidate = np.arange(5, degree_max, 5, dtype = int)

    n = len(Mass)
    
    print('Running cross validation to estimate the number of degrees of freedom for the weights. Max candidate = {}'.format(degree_max)) 
    rand_gen = np.random.choice(n, n, replace = False)
    row_size = np.int(np.floor(n/k_fold))
    a = np.arange(n)
    indices_folded = [a[i*row_size:(i+1)*row_size] if i is not k_fold-1 else a[i*row_size:] for i in range(k_fold) ]

    cv_input = ((i,j, indices_folded,n, rand_gen, Mass, Radius, Radius_sigma, Mass_sigma, 
    Log, abs_tol, location, Mass_bounds, Radius_bounds) for i in range(k_fold)for j in degree_candidate)

    pool = Pool(processes = cores)
    
    # Map the inputs to the cross validation function. Then convert to numpy array and split in k_fold separate arrays
    cv_result = list(pool.imap(cv_parallel_fn,cv_input))
    likelihood_matrix = np.split(np.array(cv_result) , k_fold)        
    likelihood_per_degree = np.sum(likelihood_matrix, axis = 0)

    print(likelihood_per_degree)
    np.savetxt(os.path.join(location,'likelihood_per_degree.txt'),likelihood_per_degree)
    deg_choose = degree_candidate[np.argmax(likelihood_per_degree)]

    print('Finished CV. Picked {} degrees by maximizing likelihood'.format({deg_choose}))


    return (deg_choose)

    

def MLE_fit_bootstrap(Mass, Radius, Mass_sigma, Radius_sigma, Mass_max = None, Mass_min = None, Radius_max = None, Radius_min = None, 
                    degree_max = 60, select_deg = 55, Log = False, k_fold = 10, num_boot = 100, 
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
        degree_max: INTEGER the maximum degree used for cross-validation/AIC/BIC. Default = 60. Suggested value = n/log10(n)
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
    print('Started for {} degrees at {}, using {} cores'.format(select_deg, starttime, cores))

    if not os.path.exists(location):
        os.mkdir(location)   
    
    with open(os.path.join(location,'log_file.txt'),'a') as f:
       f.write('Started for {} degrees at {}, using {} cores'.format(select_deg, starttime, cores))

    
    copyfile(os.path.join(os.path.dirname(location),os.path.basename(__file__)), os.path.join(location,os.path.basename(__file__)))
    copyfile(os.path.join(os.path.dirname(location),'MLE_fit.py'), os.path.join(location,'MLE_fit.py'))
    
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
        
        deg_choose = run_cross_validation(Mass = Mass, Radius = Radius, Mass_sigma = Mass_sigma, Radius_sigma = Radius_sigma,
                                        Mass_bounds = Mass_bounds, Radius_bounds = Radius_bounds, Log = Log,
                                        degree_max = degree_max, k_fold = k_fold, cores = cores, location = location, abs_tol = abs_tol)

        

        with open(os.path.join(location,'log_file.txt'),'a') as f:
            f.write('Finished CV. Picked {} degrees by maximizing likelihood'.format({deg_choose})) 
                   
    elif select_deg == 'aic' : 
        aic = np.array([MLE_fit(Mass = Mass, Radius = Radius, Mass_sigma = Mass_sigma, Radius_sigma = Radius_sigma, 
                        Mass_bounds = Mass_bounds, Radius_bounds = Radius_bounds, Log = Log, deg = d, abs_tol = abs_tol, location = location)['aic'] for d in range(2,degree_max)])
        deg_choose = np.nanmin(aic)

    elif select_deg == 'bic':
        bic = np.array([MLE_fit(Mass = Mass, Radius = Radius, Mass_sigma = Mass_sigma, Radius_sigma = Radius_sigma,
                        Mass_bounds = Mass_bounds, Radius_bounds = Radius_bounds, Log = Log, deg = d, abs_tol = abs_tol, location = location)['bic'] for d in range(2,degree_max)])
        deg_choose = np.nanmin(bic)   
                 
    elif isinstance(select_deg, (int,float)):
        deg_choose = select_deg
    
    else : 
        print('Error: Incorrect input for select_deg')
            
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
    
    print('Running {} bootstraps for the MLE code with degree = {}, using {} threads.'.format(str(num_boot),str(deg_choose),str(cores)))

    with open(os.path.join(location,'log_file.txt'),'a') as f:
       f.write('Running {} bootstraps for the MLE code with degree = {}, using {} threads.'.format(str(num_boot),str(deg_choose),str(cores)))

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
    np.savetxt(os.path.join(location,'M_points_boot.txt'),M_points_boot[0])
    np.savetxt(os.path.join(location,'R_points_boot.txt'),R_points_boot[0])    
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
            
            
if __name__ == '__main__':           
    a = MLE_fit_bootstrap(Mass = M_obs, Radius = R_obs, Mass_sigma = M_sigma, Radius_sigma = R_sigma, Mass_max = Mass_max, 
                        Mass_min = Mass_min, Radius_max = Radius_max, Radius_min = Radius_min, select_deg = 'cv', Log = True, num_boot = 60, cores = 40, 
                        location = os.path.join(os.path.dirname(__file__),'Cross_validation_40'))

            
            
        
        
        
        
    
    
    

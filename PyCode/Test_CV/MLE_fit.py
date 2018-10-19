#%cd "C:/Users/shbhu/Documents/Git/Py_mass_radius_working/PyCode"

import numpy as np
from scipy.stats import beta,norm
from scipy.integrate import quad
from scipy.optimize import brentq as root
from astropy.table import Table
from scipy.optimize import minimize, fmin_slsqp, fmin_l_bfgs_b
import datetime,os
  
    
'''
t = Table.read('MR_Kepler_170605_noanalytTTV_noupplim.csv')
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
#select_deg = 5

data = np.vstack((M_obs,R_obs)).T
sigma = np.vstack((M_sigma,R_sigma)).T

bounds = np.array([Mass_max,Mass_min,Radius_max,Radius_min])
Log = True
#deg = 5

'''

'''
y = np.log10(5)
Radius_min = -0.3 
Radius_max = 1.357509
Mass_min = -1
Mass_max = 3.809597 
qtl = [0.16, 0.84]
y_std = R_sigma
y_max = Radius_max
y_min = Radius_min
x_max = Mass_max
x_min = Mass_min
deg = 55
w_hat = weights_mle
'''
#deg = 5




def pdfnorm_beta(x, x_obs, x_sd, x_max, x_min, shape1, shape2, Log = True):
    '''
    Product of normal and beta distribution
    '''

    if Log == True:
        norm_beta = norm.pdf(x_obs, loc = 10**x, scale = x_sd) * beta.pdf((x - x_min)/(x_max - x_min), a = shape1, b = shape2)/(x_max - x_min)
    else:
        norm_beta = norm.pdf(x_obs, loc = x, scale = x_sd) * beta.pdf((x - x_min)/(x_max - x_min), a = shape1, b = shape2)/(x_max - x_min)     
        
    return norm_beta   
 
def integrate_function(data, data_sd, deg, degree, x_max, x_min, Log = False, abs_tol = 1e-10):
    '''
    Integrate the product of the normal and beta distribution
    Comment about absolute tolerance ............................ (data set specific)
    '''
    
    x_obs = data
    x_sd = data_sd
    shape1 = degree
    shape2 = deg - degree + 1
    Log = Log
                   
    return quad(pdfnorm_beta, a = x_min, b = x_max,
                 args = (x_obs, x_sd, x_max, x_min, shape1, shape2, Log), epsabs = abs_tol)[0]   
    

def find_indv_pdf(x,deg,deg_vec,x_max,x_min,x_std = None, abs_tol = 1e-10, Log = True):
    
    if x_std == None:
        x_std = (x - x_min)/(x_max - x_min)
        x_beta_indv = np.array([beta.pdf(x_std, a = d, b = deg - d + 1)/(x_max - x_min) for d in deg_vec])
    else: 
        x_beta_indv = np.array([integrate_function(data = x, data_sd = x_std, deg = deg, degree = d, x_max = x_max, x_min = x_min, abs_tol = abs_tol, Log = Log) for d in deg_vec])  
    
    return x_beta_indv
       

def marginal_density(x, x_max, x_min, deg, w_hat):
    '''
    Calculate the marginal density 
    '''
    
    if type(x) == list:
        x = np.array(x)
    
    deg_vec = np.arange(1,deg+1)    
    x_beta_indv = find_indv_pdf(x,deg,deg_vec,x_max,x_min) 
    x_beta_pdf = np.kron(x_beta_indv, np.repeat(1,deg))
    
    marg_x = np.sum(w_hat * x_beta_pdf)
    
    return marg_x
    
    
def conditional_density(y, y_max, y_min, x, x_max, x_min, deg, w_hat, abs_tol = 1e-10):
    '''
    Calculate the conditional density
    '''
    if type(x) == list:
        x = np.array(x)
    if type(y) == list:
        y = np.array(y)
    
    deg_vec = np.arange(1,deg+1)  
    
    # Return Conditional Mean, Variance, Quantile, Distribution
    y_beta_indv = find_indv_pdf(y,deg,deg_vec,y_max,y_min, abs_tol = abs_tol) 
    y_beta_pdf = np.kron(y_beta_indv, np.repeat(1,deg))
    
    denominator = np.sum(w_hat * y_beta_pdf)
    
    ########### Density ##########
    density_indv_pdf = find_indv_pdf(x,deg,deg_vec,x_max,x_min, abs_tol = abs_tol) 
    density_pdf = w_hat * np.kron(density_indv_pdf,y_beta_indv)
    
    density = density_pdf / denominator
    
    return density
    
def cond_density_quantile(y, y_max, y_min, x_max, x_min, deg, w_hat, y_std = None, qtl = [0.16,0.84], abs_tol = 1e-10):
    '''
    Calculate 16% and 84% quantiles of a conditional density, along with the mean and variance.
        
    '''   

    if type(y) == list:
        y = np.array(y)
    deg_vec = np.arange(1,deg+1)  
    
    y_beta_indv = find_indv_pdf(x = y, deg = deg, deg_vec = deg_vec, x_max = y_max, x_min = y_min, x_std = y_std, abs_tol = abs_tol) 
    y_beta_pdf = np.kron(np.repeat(1,deg),y_beta_indv)  
    denominator = np.sum(w_hat * y_beta_pdf) 
        
    # Mean
    mean_beta_indv = (deg_vec * (x_max - x_min) / (deg + 1)) + x_min
    mean_beta = np.kron(mean_beta_indv,y_beta_indv)
    mean_numerator = np.sum(w_hat * mean_beta)
    mean = mean_numerator / denominator
    
    # Variance 
    var_beta_indv = (deg_vec * (deg - deg_vec + 1) * (x_max - x_min)**2 / ((deg + 2)*(deg + 1)**2)) 
    var_beta = np.kron(var_beta_indv,y_beta_indv)
    var_numerator = np.sum(w_hat * var_beta)
    var = var_numerator / denominator
    
    # Quantile
    
    def pbeta_conditional_density(j):
    
        x_indv_cdf = np.array([beta.cdf((j - x_min)/(x_max - x_min), a = d, b = deg - d + 1) for d in deg_vec])
    
        quantile_numerator = np.sum(w_hat * np.kron(x_indv_cdf,y_beta_indv))
        p_beta = quantile_numerator / denominator
    
        return p_beta
    
                        
    def conditional_quantile(q):
        def g(x):
            return pbeta_conditional_density(x) - q  
        return root(g,a = x_min, b = x_max)
 
    quantile = [conditional_quantile(i) for i in qtl]
    
    #print(mean,var,quantile,denominator,y_beta_indv)
    

    
    return mean, var, quantile[0], quantile[1], denominator, y_beta_indv

def mixture_conditional_density_qtl(y_max, y_min, x_max, x_min, deg, w_hat, denominator_sample, y_beta_indv_sample,qtl = [0.16,0.84]):
    '''   
    Calculate the 16% and 84% quantiles using root function.
    Mixture of the CDF of k conditional densities
    '''
    
    deg_vec = np.arange(1,deg+1)  

    def pbeta_conditional_density(j):
        
        x_indv_cdf = np.array([beta.cdf((j - x_min)/(x_max - x_min), a = d, b = deg - d + 1) for d in deg_vec])
        quantile_numerator = np.zeros(len(denominator_sample))
        p_beta = np.zeros(len(denominator_sample))

        for i in range(0,len(denominator_sample)):
            quantile_numerator[i] = np.sum(w_hat * np.kron(x_indv_cdf,y_beta_indv_sample[i]))
            p_beta[i] = quantile_numerator[i] / denominator_sample[i]
        
        return np.mean(p_beta)
           
    def conditional_quantile(q):
        def g(x):
            return pbeta_conditional_density(x) - q   
        return root(g,a = x_min, b = x_max)

    quantile = [conditional_quantile(i) for i in qtl]
    
    return quantile
    

def calc_C_matrix(n, deg, M, Mass_sigma, M_max, M_min, R, Radius_sigma, R_max, R_min, Log, abs_tol, location):
    '''
    
    
    
    '''
    
    deg_vec = np.arange(1,deg+1) 
    
    if Mass_sigma is None:
        # pdf for Mass and Radius for each beta density
        M_indv_pdf = find_indv_pdf(M, deg, deg_vec, M_max, M_min, x_std = None, abs_tol = abs_tol) 
        R_indv_pdf = find_indv_pdf(R, deg, deg_vec, R_max, R_min, x_std = None, abs_tol = abs_tol) 
        
    else:        
        M_indv_pdf = np.zeros((n, deg))
        R_indv_pdf = np.zeros((n, deg))
        C_pdf = np.zeros((n, deg**2))
        
        print('Started Integration at ',datetime.datetime.now())
        with open(os.path.join(location,'log_file.txt'),'a') as f:
            f.write('Started Integration at {}\n'.format(datetime.datetime.now()))
        for i in range(0,n): 
            M_indv_pdf[i,:] = find_indv_pdf(M[i], deg, deg_vec, M_max, M_min, Mass_sigma[i], abs_tol, Log = Log)  
            R_indv_pdf[i,:] = find_indv_pdf(R[i], deg, deg_vec, R_max, R_min, Radius_sigma[i], abs_tol, Log = Log)
                                  
            # put M.indv.pdf and R.indv.pdf into a big matrix
            C_pdf[i,:] = np.kron(M_indv_pdf[i], R_indv_pdf[i])
            
        C_pdf = C_pdf.T 
    
    return C_pdf

    
######################################
##### Main function: MLE.fit #########
######################################

#a = MLE_fit(data = data, bounds = bounds, deg = 55, sigma = sigma, output_weights_only = False, Log = True)

def MLE_fit(Mass, Radius, Mass_sigma, Radius_sigma, Mass_bounds, Radius_bounds,
            deg, Log = False,abs_tol = 1e-10, output_weights_only = False, location = None):
    '''
    INPUT:
        Mass: Mass measurements 
        Radius: Radius measurements
        Mass_sigma: Mass measurement errors
        Radius_sigma: Radius measurement errors
        Mass_bounds: Mass upper and lower bounds
        Radius_bounds: Radius upper and lower bounds
        deg: Degree used for Bernstein polynomials
        Log: If True, data is transformed into Log scale
        abs_tol: Precision used to calculate integral
        output_weights_only: If True, only output the estimated weights from 
        the Bernstein polynomials. Else, output the conditional densities
        location : For logging

    '''
    
    print('New MLE')
    starttime = datetime.datetime.now()
    if location is None:
        location = os.path.dirname(__file__)
    with open(os.path.join(location,'log_file.txt'),'a') as f:
       f.write('\n======================================\n')
       f.write('Started run at {}\n'.format(starttime))


    
    # Read the Data
    
    n = np.shape(Mass)[0]

        
    M_max = Mass_bounds[0]
    M_min = Mass_bounds[1]
    R_max = Radius_bounds[0]
    R_min = Radius_bounds[1]
    
    
    
    
 
    """    
    if Mass_sigma is None:
        # pdf for Mass and Radius for each beta density
        M_indv_pdf = np.array([beta.pdf((Mass - M_min)/(M_max - M_min), a = d, b = deg - d+1)/(M_max - M_min) for d in deg_vec])
        R_indv_pdf = np.array([beta.pdf((Radius - R_min)/(R_max - R_min), a = d, b = deg - d+1)/(R_max - R_min) for d in deg_vec])        
        
    else:        
        M_indv_pdf = np.zeros((n,deg))
        R_indv_pdf = np.zeros((n,deg))
        C_pdf = np.zeros((n,deg**2))
        
        print('Started Integration at ',datetime.datetime.now())
        with open(os.path.join(location,'log_file.txt'),'a') as f:
            f.write('Started Integration at {}\n'.format(datetime.datetime.now()))

        for i in range(0,n):  
            print(Mass[i],deg,deg_vec,M_max,M_min,Mass_sigma[i],Log)
 
            M_indv_pdf[i,:] = find_indv_pdf(Mass[i], deg, deg_vec, M_max, M_min, Mass_sigma[i], abs_tol, Log)
  
            R_indv_pdf[i,:] = find_indv_pdf(Radius[i], deg, deg_vec, R_max, R_min, Radius_sigma[i], abs_tol, Log)
            
            '''                                  
            for d in deg_vec:                
                a = datetime.datetime.now()
                
                # pdf for Mass for integrated beta density and normal density
                M_indv_pdf[i,d-1] = integrate_function(data = Mass[i], data_sd = Mass_sigma[i], 
                                    deg = deg, degree = d , x_max = M_max, x_min = M_min, Log = Log, abs_tol = abs_tol)
                # pdf for Radius for integrated beta density and normal density
                R_indv_pdf[i,d-1] = integrate_function(data = Radius[i], data_sd = Radius_sigma[i], 
                                    deg = deg, degree = d , x_max = R_max, x_min = R_min, Log = Log, abs_tol = abs_tol)
                b = datetime.datetime.now()
            # put M.indv.pdf and R.indv.pdf into a big matrix
            '''

            C_pdf[i,:] = np.kron(M_indv_pdf[i],R_indv_pdf[i])

            
        C_pdf = C_pdf.T 
    """
    
    C_pdf = calc_C_matrix(n = n, deg = deg, M = Mass, Mass_sigma = Mass_sigma, M_max = M_max, M_min = M_min,
                        R = Radius, Radius_sigma = Radius_sigma, R_max = R_max, R_min = R_min, Log = Log, abs_tol = abs_tol, location = location)

    print(np.shape(C_pdf))
    np.savetxt('C_pdf.txt',C_pdf)

    print('Finished Integration at ',datetime.datetime.now())
    with open(os.path.join(location,'log_file.txt'),'a') as f:
        f.write('Finished Integration at {}\n'.format(datetime.datetime.now()))

    print('Calculated the PDF for Mass and Radius for Integrated Beta and Normal Density')

        
    # Function input to optimizer            
    def fn1(w):
        # Log of 0 throws weird errors
        C_pdf[C_pdf == 0] = 1e-300
        
        a = - np.sum(np.log(np.matmul(w,C_pdf)))
        return a
        
    def eqn(w):
        return np.sum(w) - 1
        
    def fn2(w):
        # Log of 0 throws weird errors
        C_pdf[C_pdf == 0] = 1e-300
        
        w[-1] = 1- np.sum(w[0:-1])
        
        a = - np.sum(np.log(np.matmul(w,C_pdf)))
        return a

    bounds = [[0,1]]*deg**2
    x0 = np.repeat(1./(deg**2),deg**2)
    #print('Using slsqp with bigger steps')
    #opt_result = fmin_slsqp(fn2, x0, bounds = bounds, iter = 1e3, full_output = True, iprint = 1)
    opt_result = fmin_slsqp(fn1, x0, bounds = bounds, f_eqcons = eqn, iter = 500,full_output = True, iprint = 1, epsilon = 1e-5,acc = 1e-5)
    print('Optimization run finished at {}, with {} iterations. Exit Code = {}\n\n'.format(datetime.datetime.now(),opt_result[2],opt_result[3],opt_result[4]))

    with open(os.path.join(location,'log_file.txt'),'a') as f:
        f.write('Finished Optimization at {}'.format(datetime.datetime.now()))
        f.write('Optimization terminated after {} iterations. Exit Code = {}{}\n\n'.format(opt_result[2],opt_result[3],opt_result[4]))

    w_hat = opt_result[0]
    n_log_lik = opt_result[1]
    
    # Calculate AIC and BIC    
    aic = n_log_lik*2 + 2*(deg**2 - 1)
    bic = n_log_lik*2 + np.log(n)*(deg**2 - 1)
    
    
    # marginal densities
    M_seq = np.linspace(M_min,M_max,100)
    R_seq = np.linspace(R_min,R_max,100)
    Mass_marg = np.array([marginal_density(x = m, x_max = M_max, x_min = M_min, deg = deg, w_hat = w_hat) for m in M_seq])
    Radius_marg = np.array([marginal_density(x = r, x_max = R_max, x_min = R_min, deg = deg, w_hat = w_hat) for r in R_seq])
        
    output = {'weights':w_hat,'aic':aic,'bic':bic,'M_points':M_seq,'R_points':R_seq,
                  'Mass_marg':Mass_marg,'Radius_marg':Radius_marg}

    # conditional densities with 16% and 84% quantile

    M_cond_R = np.array([cond_density_quantile(y = r, y_max = R_max, y_min = R_min,
                        x_max = M_max, x_min = M_min, deg = deg, w_hat = w_hat, qtl = [0.16,0.84])[0:4] for r in R_seq])
    
        
    M_cond_R_mean = M_cond_R[:,0]
    M_cond_R_var = M_cond_R[:,1]
    M_cond_R_quantile = M_cond_R[:,2:4]
    
    R_cond_M = np.array([cond_density_quantile(y = m, y_max = M_max, y_min = M_min,
                        x_max = R_max, x_min = R_min, deg = deg, w_hat = np.reshape(w_hat,(deg,deg)).T.flatten(), qtl = [0.16,0.84])[0:4] for m in M_seq])
    R_cond_M_mean = R_cond_M[:,0]
    R_cond_M_var = R_cond_M[:,1]
    R_cond_M_quantile = R_cond_M[:,2:4]
    
    output['M_cond_R'] = M_cond_R_mean
    output['M_cond_R_var'] = M_cond_R_var
    output['M_cond_R_quantile'] = M_cond_R_quantile
    output['R_cond_M'] = R_cond_M_mean
    output['R_cond_M_var'] = R_cond_M_var
    output['R_cond_M_quantile'] = R_cond_M_quantile
    
    if output_weights_only == True:
        return w_hat
    else:
        return output
    

  
    

        
    


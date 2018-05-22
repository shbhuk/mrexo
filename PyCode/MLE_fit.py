#%cd "C:/Users/shbhu/Documents/Git/Py_mass_radius_working/PyCode"

import numpy as np
from scipy.stats import beta,norm
from scipy.integrate import quad
from scipy.optimize import brentq as root
from astropy.table import Table
from scipy.optimize import minimize, fmin_slsqp
import datetime
    
    

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

abs_tol = 1e-20
Log = True



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
       

def marginal_density(x, x_max, x_min, deg, w_hat):
    '''
    Calculate the marginal density 
    '''
    
    if type(x) == list:
        x = np.array(x)
    
    x_std = (x - x_min)/(x_max - x_min)
    deg_vec = np.arange(1,deg+1)    
    x_beta_indv = np.array([beta.pdf(x_std, a = d, b = deg - d + 1)/(x_max - x_min) for d in deg_vec])
    
    x_beta_pdf = np.kron(x_beta_indv, np.repeat(1,deg))
    
    marg_x = np.sum(w_hat * x_beta_pdf)
    
    return marg_x
    
    
def conditional_density(y, y_max, y_min, x, x_max, x_min, deg, w_hat):
    '''
    Calculate the conditional density
    '''
    if type(x) == list:
        x = np.array(x)
    if type(y) == list:
        y = np.array(y)
    
    deg_vec = np.arange(1,deg+1)  
    
    # Return Conditional Mean, Variance, Quantile, Distribution
    y_std = (y - y_min)/(y_max - y_min)
    x_std = (x - x_min)/(x_max - x_min)
    y_beta_indv = np.array([beta.pdf(y_std, a = d, b = deg - d + 1)/(y_max - y_min) for d in deg_vec])
    y_beta_pdf = np.kron(y_beta_indv, np.repeat(1,deg))
    
    denominator = np.sum(w_hat * y_beta_pdf)
    
    ########### Density ##########
    
    density_indv_pdf = np.array([beta.pdf(x_std, a = d, b = deg - d + 1)/(x_max - x_min) for d in deg_vec])
    density_pdf = w_hat * np.kron(density_indv_pdf,y_beta_indv)
    
    density = density_pdf / denominator
    
    return density
    
def cond_density_quantile(y, y_max, y_min, x_max, x_min, deg, w_hat, qtl = [0.16,0.84]):
    '''
    Calculate 16% and 84% quantiles of a conditional density
        
    '''   

    if type(y) == list:
        y = np.array(y)
    
    deg_vec = np.arange(1,deg+1)  
    
    y_std = (y - y_min)/(y_max - y_min)
    y_beta_indv = np.array([beta.pdf(y_std, a = d, b = deg - d + 1)/(y_max - y_min) for d in deg_vec])
    y_beta_pdf = np.kron(np.repeat(1,deg),y_beta_indv)
    
    denominator = np.sum(w_hat * y_beta_pdf)
    
    # Mean
    mean_beta_indv = (deg_vec * (x_max - x_min) / (deg + 1)) + x_min
    mean_beta = np.kron(mean_beta_indv,y_beta_indv)
    mean_nominator = np.sum(w_hat * mean_beta)
    mean = mean_nominator / denominator
    
    # Variance 
    var_beta_indv = (deg_vec * (deg - deg_vec + 1) * (x_max - x_min)**2 / ((deg + 2)*(deg + 1)**2)) 
    var_beta = np.kron(var_beta_indv,y_beta_indv)
    var_nominator = np.sum(w_hat * var_beta)
    var = var_nominator / denominator
    
    # Quantile
    
    def pbeta_conditional_density(x):
        
        def mix_density(j):
            
            x_indv_cdf = np.array([beta.cdf((j - x_min)/(x_max - x_min), a = d, b = deg - d + 1) for d in deg_vec])

            quantile_nominator = np.sum(w_hat * np.kron(x_indv_cdf,y_beta_indv))
            p_beta = quantile_nominator / denominator
            
            return p_beta

        if np.size(x)>1:
            return np.array([mix_density(i) for i in x])
        else:
            return mix_density(x)
            
            
    def conditional_quantile(q):
        def g(x):
            return pbeta_conditional_density(x) - q
        return root(g,a = x_min, b = x_max)
 
    quantile = [conditional_quantile(i) for i in qtl]
    
    return mean, var, quantile[0], quantile[1]
    
    
######################################
##### Main function: MLE.fit #########
######################################

#a = MLE_fit(data = data, bounds = bounds, deg = deg, sigma = sigma, output_weights_only = False, Log = True)


def MLE_fit(data, bounds, deg, sigma = None, Log = False,
                    abs_tol = 1e-20, output_weights_only = False):
    '''
    INPUT:
        data: The first column contains the mass measurements and 
              the second column contains the radius measurements.
              Numpy Array
        sigma: Measurement Errors for the Data. Default is None
        bounds: Vector with 4 elements. Upper and lower bound for Mass, Upper and lower bound for Radius.
        deg: Degree used for Bernstein polynomials
        Log: If True, data is transformed into Log scale
        abs_tol: Precision used to calculate integral
        output_weights_only: If True, only output the estimated weights from 
        the Bernstein polynomials. Else, output the conditional densities

    '''

    
    if np.shape(data)[0] < np.shape(data)[1]:
        data = np.transpose(data)
    
    # Read the Data
    
    n = np.shape(data)[0]
    M = data[:,0]
    R = data[:,1]
      
    if sigma is not None:
        sigma_M = sigma[:,0]
        sigma_R = sigma[:,1]
        
    M_max = bounds[0]
    M_min = bounds[1]
    R_max = bounds[2]
    R_min = bounds[3]
    
    deg_vec = np.arange(1,deg+1)     
    
    if sigma is None:
        # pdf for Mass and Radius for each beta density
        M_indv_pdf = np.array([beta.pdf((M - M_min)/(M_max - M_min), a = d, b = deg - d+1)/(M_max - M_min) for d in deg_vec])
        R_indv_pdf = np.array([beta.pdf((R - R_min)/(R_max - R_min), a = d, b = deg - d+1)/(R_max - R_min) for d in deg_vec])        
        
    else:        
        M_indv_pdf = np.zeros((n,deg))
        R_indv_pdf = np.zeros((n,deg))
        C_pdf = np.zeros((n,deg**2))
        
        print(datetime.datetime.now())
        for i in range(0,n):        
            for d in deg_vec:
                # pdf for Mass for integrated beta density and normal density
                M_indv_pdf[i,d-1] = integrate_function(data = M[i], data_sd = sigma_M[i], 
                                    deg = deg, degree = d , x_max = M_max, x_min = M_min, Log = Log, abs_tol = abs_tol)
                # pdf for Radius for integrated beta density and normal density
                R_indv_pdf[i,d-1] = integrate_function(data = R[i], data_sd = sigma_R[i], 
                                    deg = deg, degree = d , x_max = R_max, x_min = R_min, Log = Log, abs_tol = abs_tol)

            # put M.indv.pdf and R.indv.pdf into a big matrix
            C_pdf[i,:] = np.kron(M_indv_pdf[i],R_indv_pdf[i])

            
        C_pdf = C_pdf.T 
    print(datetime.datetime.now())
    print('Calculated the PDF for Mass and Radius for Integrated Beta and Normal Density')
     
    test = []     
    # Function input to optimizer            
    def fn1(w):
        # Log of 0 throws weird errors
        C_pdf[C_pdf == 0] = 1e-300
        
        a = - np.sum(np.log(np.matmul(w,C_pdf)))
        test.append(a)
        return a
  
        
    def eqn(w):
        return np.sum(w) - 1

    bounds = [[0,1]]*deg**2
    x0 = np.repeat(1./(deg**2),deg**2)
    
    opt_result = fmin_slsqp(fn1, x0, bounds = bounds, f_eqcons=eqn,iter=1e3,full_output = True, iprint = 0)
    print('Optimization run')
    
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
                        x_max = M_max, x_min = M_min, deg = deg, w_hat = w_hat, qtl = [0.16,0.84]) for r in R_seq])
    
        
    M_cond_R_mean = M_cond_R[:,0]
    M_cond_R_var = M_cond_R[:,1]
    M_cond_R_quantile = M_cond_R[:,2:4]
    
    R_cond_M = np.array([cond_density_quantile(y = m, y_max = M_max, y_min = M_min,
                        x_max = R_max, x_min = R_min, deg = deg, w_hat = np.reshape(w_hat,(deg,deg)).T.flatten(), qtl = [0.16,0.84]) for m in M_seq])
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
    

    

  
    

        
    


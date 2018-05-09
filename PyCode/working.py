import numpy as np
from scipy.stats import beta,norm
from scipy.integrate import quad
from scipy.optimize import brentq as root


x_max = 5
x_min = 0.01
x = 0.1
y = 0.17
y_max = 11
y_min = 0.03
w_hat = 1
deg = 5


def pdfnorm_beta(x, x_obs, x_sd, x_max, x_min, shape1, shape2, log = True):
    '''
    Product of normal and beta distribution
    '''
    if log == True:
        norm_beta = norm.pdf(x_obs, loc = 10**x, scale = x_sd) * beta.pdf((x - x_min)/(x_max - x_min), a = shape1, b = shape2)/(x_max - x_min)
    else:
        norm_beta = norm.pdf(x_obs, loc = x, scale = x_sd) * beta.pdf((x - x_min)/(x_max - x_min), a = shape1, b = shape2)/(x_max - x_min)     
        
    return norm_beta   
 
def integrate_function(data, deg, degree, x_max, x_min, log = False, abs_tol = 1e-10):
    '''
    Integrate the product of the normal and beta distribution
    '''
    
    x_obs = data[0]
    x_sd = data[1]
    shape1 = degree
    shape2 = deg - degree + 1
    log = log
    
    return quad(pdfnorm_beta, a = x_min, b = x_max,
                 args = (x_obs, x_sd, x_max, x_min, shape1, shape2, log), epsabs = abs_tol)[0]
       

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
    y_beta_pdf = np.kron(y_beta_indv, np.repeat(1,deg))
    
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
    
    return [mean,var,quantile]
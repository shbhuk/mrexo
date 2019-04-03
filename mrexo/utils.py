import numpy as np
import os
import sys
if sys.version_info.major==3:
    from functools import lru_cache
else:
    from functools32 import lru_cache

def save_dictionary(dictionary, output_location, bootstrap=False):
    """
    Save the keys in the dictionary as separate data .txt files.

    INPUTS:
        dictionary : Output dictionary from fitting without bootstrap using Maximum Likelihood Estimation.
                     The keys in the dictionary are:
                            'weights' : Weights for Beta densities.
                            'aic' : Akaike Information Criterion.
                            'bic' : Bayesian Information Criterion.
                            'M_points' : Sequence of mass points for initial fitting w/o bootstrap.
                            'R_points' : Sequence of radius points for initial fitting w/o bootstrap.
                            'M_cond_R' : Conditional distribution of mass given radius.
                            'M_cond_R_var' : Variance for the Conditional distribution of mass given radius.
                            'M_cond_R_quantile' : Quantiles for the Conditional distribution of mass given radius.
                            'R_cond_M' : Conditional distribution of radius given mass.
                            'R_cond_M_var' : Variance for the Conditional distribution of radius given mass.
                            'R_cond_M_quantile' : Quantiles for the Conditional distribution of radius given mass.
                            if bootstrap == False:
                            'joint_dist' : Joint distribution of mass AND radius.
        output_location : The output subdirectory within save_path where the files are stored
        bootstrap : If False, will save files with initial fitting names. Else the files will be saved with bootstrap header and file name.

    OUTPUTS:
        Returns nothing. Saves the contents of the dictionary
    """
    aux_output_location = os.path.join(output_location, 'other_data_products')

    if bootstrap == False:
        weights = dictionary['weights']
        aic = dictionary['aic']
        bic = dictionary['bic']
        M_points =  dictionary['M_points']
        R_points = dictionary['R_points']
        M_cond_R = dictionary['M_cond_R']
        M_cond_R_var = dictionary['M_cond_R_var']
        M_cond_R_lower = dictionary['M_cond_R_quantile'][:,0]
        M_cond_R_upper = dictionary['M_cond_R_quantile'][:,1]
        R_cond_M = dictionary['R_cond_M']
        R_cond_M_var = dictionary['R_cond_M_var']
        R_cond_M_lower = dictionary['R_cond_M_quantile'][:,0]
        R_cond_M_upper = dictionary['R_cond_M_quantile'][:,1]
        joint_dist = dictionary['joint_dist']


        np.savetxt(os.path.join(output_location,'weights.txt'),weights, comments='#', header='Weights for Beta densities from initial fitting w/o bootstrap')
        np.savetxt(os.path.join(aux_output_location,'aic.txt'),[aic], comments='#', header='Akaike Information Criterion from initial fitting w/o bootstrap')
        np.savetxt(os.path.join(aux_output_location,'bic.txt'),[bic], comments='#', header='Bayesian Information Criterion from initial fitting w/o bootstrap')
        np.savetxt(os.path.join(output_location,'M_points.txt'), M_points, comments='#', header='Sequence of mass points for initial fitting w/o bootstrap')
        np.savetxt(os.path.join(output_location,'R_points.txt'), R_points, comments='#', header='Sequence of radius points for initial fitting w/o bootstrap')
        np.savetxt(os.path.join(output_location,'M_cond_R.txt'), M_cond_R, comments='#', header='Conditional distribution of mass given radius from initial fitting w/o bootstrap')
        np.savetxt(os.path.join(aux_output_location,'M_cond_R_var.txt'), M_cond_R_var, comments='#', header='Variance for the Conditional distribution of mass given radius from initial fitting w/o bootstrap')
        np.savetxt(os.path.join(output_location,'M_cond_R_lower.txt'), M_cond_R_lower, comments='#', header='Lower limit for the Conditional distribution of mass given radius from initial fitting w/o bootstrap')
        np.savetxt(os.path.join(output_location,'M_cond_R_upper.txt'), M_cond_R_upper, comments='#', header='Upper limit for the Conditional distribution of mass given radius from initial fitting w/o bootstrap')
        np.savetxt(os.path.join(output_location,'R_cond_M.txt'), R_cond_M, comments='#', header='Conditional distribution of radius given mass from initial fitting w/o bootstrap')
        np.savetxt(os.path.join(aux_output_location,'R_cond_M_var.txt'), R_cond_M_var, comments='#', header='Variance for the Conditional distribution of radius given mass from initial fitting w/o bootstrap')
        np.savetxt(os.path.join(output_location,'R_cond_M_lower.txt'), R_cond_M_lower, comments='#', header='Lower limit for the Conditional distribution of radius given mass from initial fitting w/o bootstrap')
        np.savetxt(os.path.join(output_location,'R_cond_M_upper.txt'), R_cond_M_upper, comments='#', header='Upper limit for the Conditional distribution of radius given mass from initial fitting w/o bootstrap')
        np.savetxt(os.path.join(output_location,'joint_distribution.txt'), joint_dist, comments='#', header='Joint distribution of mass and radius w/o bootstrap')

    else:
        weights_boot = np.array([x['weights'] for x in dictionary])
        aic_boot = np.array([x['aic'] for x in dictionary])
        bic_boot = np.array([x['bic'] for x in dictionary])
        M_points_boot =  np.array([x['M_points'] for x in dictionary])
        R_points_boot = np.array([x['R_points'] for x in dictionary])
        M_cond_R_boot = np.array([x['M_cond_R'] for x in dictionary])
        M_cond_R_var_boot = np.array([x['M_cond_R_var'] for x in dictionary])
        M_cond_R_lower_boot = np.array([x['M_cond_R_quantile'][:,0] for x in dictionary])
        M_cond_R_upper_boot = np.array([x['M_cond_R_quantile'][:,1] for x in dictionary])
        R_cond_M_boot = np.array([x['R_cond_M'] for x in dictionary])
        R_cond_M_var_boot = np.array([x['R_cond_M_var'] for x in dictionary])
        R_cond_M_lower_boot = np.array([x['R_cond_M_quantile'][:,0] for x in dictionary])
        R_cond_M_upper_boot = np.array([x['R_cond_M_quantile'][:,1] for x in dictionary])

        np.savetxt(os.path.join(output_location,'weights_boot.txt'),weights_boot, comments='#', header='Weights for Beta densities from bootstrap run')
        np.savetxt(os.path.join(aux_output_location,'aic_boot.txt'),aic_boot, comments='#', header='Akaike Information Criterion from bootstrap run')
        np.savetxt(os.path.join(aux_output_location,'bic_boot.txt'),bic_boot, comments='#', header='Bayesian Information Criterion from bootstrap run')
        np.savetxt(os.path.join(aux_output_location,'M_points_boot.txt'),M_points_boot, comments='#', header='Sequence of mass points for bootstrap run')
        np.savetxt(os.path.join(aux_output_location,'R_points_boot.txt'),R_points_boot, comments='#', header='Sequence of radius points for bootstrap run')
        np.savetxt(os.path.join(output_location,'M_cond_R_boot.txt'),M_cond_R_boot, comments='#', header='Conditional distribution of mass given radius from bootstrap run')
        np.savetxt(os.path.join(aux_output_location,'M_cond_R_var_boot.txt'),M_cond_R_var_boot, comments='#', header='Variance for the Conditional distribution of mass given radius from bootstrap run')
        np.savetxt(os.path.join(aux_output_location,'M_cond_R_lower_boot.txt'),M_cond_R_lower_boot, comments='#', header='Lower limit for the Conditional distribution of mass given radius from bootstrap run')
        np.savetxt(os.path.join(aux_output_location,'M_cond_R_upper_boot.txt'),M_cond_R_upper_boot, comments='#', header='Upper limit for the Conditional distribution of mass given radius from bootstrap run')
        np.savetxt(os.path.join(output_location,'R_cond_M_boot.txt'),R_cond_M_boot, comments='#', header='Conditional distribution of radius given mass from bootstrap run')
        np.savetxt(os.path.join(aux_output_location,'R_cond_M_var_boot.txt'),R_cond_M_var_boot, comments='#', header='Variance for the Conditional distribution of radius given mass from bootstrap run')
        np.savetxt(os.path.join(aux_output_location,'R_cond_M_lower_boot.txt'),R_cond_M_lower_boot, comments='#', header='Lower limit for the Conditional distribution of radius given mass from bootstrap run')
        np.savetxt(os.path.join(aux_output_location,'R_cond_M_upper_boot.txt'),R_cond_M_upper_boot, comments='#', header='Upper limit for the Conditional distribution of radius given mass from bootstrap run')


@lru_cache(maxsize=200)
def load_lookup_table(f_path):
    """
    Load the lookup table interpolate object and pass the object.
    INPUT:
        f_path : Entire file path for the .npy interpolated file.
    OUTPUT:
        lookup_inter : Interpolated lookup table (.npy) object.

    """

    lookup_inter = np.load(f_path, encoding = 'bytes').item()
    print('Loaded lookup table from {}'.format(f_path))

    return lookup_inter

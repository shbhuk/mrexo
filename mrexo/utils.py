import numpy as np
import os
from multiprocessing import current_process
import sys
if sys.version_info.major==3:
    from functools import lru_cache
else:
    from functools32 import lru_cache

def _save_dictionary(dictionary, output_location,
        X_char, Y_char, X_label, Y_label,bootstrap=False):

    """
    Save the keys in the dictionary as separate data .txt files.

    INPUTS:
        dictionary : Output dictionary from fitting without bootstrap using Maximum Likelihood Estimation.
                     The keys in the dictionary are:
                            'weights' : Weights for Beta densities.
                            'aic' : Akaike Information Criterion.
                            'bic' : Bayesian Information Criterion.
                            'Y_points' : Sequence of Y points for initial fitting w/o bootstrap.
                            'X_points' : Sequence of X points for initial fitting w/o bootstrap.
                            'Y_cond_X' : Conditional distribution of Y given X.
                            'Y_cond_X_var' : Variance for the Conditional distribution of Y given X.
                            'Y_cond_X_quantile' : Quantiles for the Conditional distribution of Y given X.
                            'X_cond_Y' : Conditional distribution of X given Y.
                            'X_cond_Y_var' : Variance for the Conditional distribution of X given Y.
                            'X_cond_Y_quantile' : Quantiles for the Conditional distribution of X given Y.
                            if bootstrap == False:
                            'joint_dist' : Joint distribution of Y AND X.
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
        Y_points =  dictionary['Y_points']
        X_points = dictionary['X_points']
        Y_cond_X = dictionary['Y_cond_X']
        Y_cond_X_var = dictionary['Y_cond_X_var']
        Y_cond_X_lower = dictionary['Y_cond_X_quantile'][:,0]
        Y_cond_X_upper = dictionary['Y_cond_X_quantile'][:,1]
        X_cond_Y = dictionary['X_cond_Y']
        X_cond_Y_var = dictionary['X_cond_Y_var']
        X_cond_Y_lower = dictionary['X_cond_Y_quantile'][:,0]
        X_cond_Y_upper = dictionary['X_cond_Y_quantile'][:,1]
        joint_dist = dictionary['joint_dist']


        np.savetxt(os.path.join(output_location,'weights.txt'),weights, comments='#', header='Weights for Beta densities from initial fitting w/o bootstrap')
        np.savetxt(os.path.join(aux_output_location,'aic.txt'),[aic], comments='#', header='Akaike Information Criterion from initial fitting w/o bootstrap')
        np.savetxt(os.path.join(aux_output_location,'bic.txt'),[bic], comments='#', header='Bayesian Information Criterion from initial fitting w/o bootstrap')
        np.savetxt(os.path.join(output_location,'Y_points.txt'), Y_points, comments='#', header='Sequence of {} points for initial fitting w/o bootstrap'.format(Y_label))
        np.savetxt(os.path.join(output_location,'X_points.txt'), X_points, comments='#', header='Sequence of {} points for initial fitting w/o bootstrap'.format(X_label))
        np.savetxt(os.path.join(output_location,'Y_cond_X.txt'), Y_cond_X, comments='#', header='Conditional distribution of {} given {} from initial fitting w/o bootstrap'.format(Y_label, X_label))
        np.savetxt(os.path.join(aux_output_location,'Y_cond_X_var.txt'), Y_cond_X_var, comments='#', header='Variance for the Conditional distribution of {} given {} from initial fitting w/o bootstrap'.format(Y_label, X_label))
        np.savetxt(os.path.join(output_location,'Y_cond_X_lower.txt'), Y_cond_X_lower, comments='#', header='Lower limit for the Conditional distribution of {} given {} from initial fitting w/o bootstrap'.format(Y_label, X_label))
        np.savetxt(os.path.join(output_location,'Y_cond_X_upper.txt'), Y_cond_X_upper, comments='#', header='Upper limit for the Conditional distribution of {} given {} from initial fitting w/o bootstrap'.format(Y_label, X_label))
        np.savetxt(os.path.join(output_location,'X_cond_Y.txt'), X_cond_Y, comments='#', header='Conditional distribution of {} given {} from initial fitting w/o bootstrap'.format(X_label, Y_label))
        np.savetxt(os.path.join(aux_output_location,'X_cond_Y_var.txt'), X_cond_Y_var, comments='#', header='Variance for the Conditional distribution of {} given {} from initial fitting w/o bootstrap'.format(X_label, Y_label))
        np.savetxt(os.path.join(output_location,'X_cond_Y_lower.txt'), X_cond_Y_lower, comments='#', header='Lower limit for the Conditional distribution of {} given {} from initial fitting w/o bootstrap'.format(X_label, Y_label))
        np.savetxt(os.path.join(output_location,'X_cond_Y_upper.txt'), X_cond_Y_upper, comments='#', header='Upper limit for the Conditional distribution of {} given {} from initial fitting w/o bootstrap'.format(X_label, Y_label))
        np.savetxt(os.path.join(output_location,'joint_distribution.txt'), joint_dist, comments='#', header='Joint distribution of {} and {} w/o bootstrap'.format(Y_label, X_label))

    else:
        weights_boot = np.array([x['weights'] for x in dictionary])
        aic_boot = np.array([x['aic'] for x in dictionary])
        bic_boot = np.array([x['bic'] for x in dictionary])
        Y_points_boot =  np.array([x['Y_points'] for x in dictionary])
        X_points_boot = np.array([x['X_points'] for x in dictionary])
        Y_cond_X_boot = np.array([x['Y_cond_X'] for x in dictionary])
        Y_cond_X_var_boot = np.array([x['Y_cond_X_var'] for x in dictionary])
        Y_cond_X_lower_boot = np.array([x['Y_cond_X_quantile'][:,0] for x in dictionary])
        Y_cond_X_upper_boot = np.array([x['Y_cond_X_quantile'][:,1] for x in dictionary])
        X_cond_Y_boot = np.array([x['X_cond_Y'] for x in dictionary])
        X_cond_Y_var_boot = np.array([x['X_cond_Y_var'] for x in dictionary])
        X_cond_Y_lower_boot = np.array([x['X_cond_Y_quantile'][:,0] for x in dictionary])
        X_cond_Y_upper_boot = np.array([x['X_cond_Y_quantile'][:,1] for x in dictionary])

        np.savetxt(os.path.join(output_location,'weights_boot.txt'),weights_boot, comments='#', header='Weights for Beta densities from bootstrap run')
        np.savetxt(os.path.join(aux_output_location,'aic_boot.txt'),aic_boot, comments='#', header='Akaike Information Criterion from bootstrap run')
        np.savetxt(os.path.join(aux_output_location,'bic_boot.txt'),bic_boot, comments='#', header='Bayesian Information Criterion from bootstrap run')
        np.savetxt(os.path.join(aux_output_location,'Y_points_boot.txt'),Y_points_boot, comments='#', header='Sequence of Y points for bootstrap run')
        np.savetxt(os.path.join(aux_output_location,'X_points_boot.txt'),X_points_boot, comments='#', header='Sequence of X points for bootstrap run')
        np.savetxt(os.path.join(output_location,'Y_cond_X_boot.txt'),Y_cond_X_boot, comments='#', header='Conditional distribution of {} given {} from bootstrap run'.format(Y_label, X_label))
        np.savetxt(os.path.join(aux_output_location,'Y_cond_X_var_boot.txt'),Y_cond_X_var_boot, comments='#', header='Variance for the Conditional distribution of {} given {} from bootstrap run'.format(Y_label, X_label))
        np.savetxt(os.path.join(aux_output_location,'Y_cond_X_lower_boot.txt'),Y_cond_X_lower_boot, comments='#', header='Lower limit for the Conditional distribution of {} given {} from bootstrap run'.format(Y_label, X_label))
        np.savetxt(os.path.join(aux_output_location,'Y_cond_X_upper_boot.txt'),Y_cond_X_upper_boot, comments='#', header='Upper limit for the Conditional distribution of {} given {} from bootstrap run'.format(Y_label, X_label))
        np.savetxt(os.path.join(output_location,'X_cond_Y_boot.txt'),X_cond_Y_boot, comments='#', header='Conditional distribution of {} given {} from bootstrap run'.format(X_label, Y_label))
        np.savetxt(os.path.join(aux_output_location,'X_cond_Y_var_boot.txt'),X_cond_Y_var_boot, comments='#', header='Variance for the Conditional distribution of {} given {} from bootstrap run'.format(X_label, Y_label))
        np.savetxt(os.path.join(aux_output_location,'X_cond_Y_lower_boot.txt'),X_cond_Y_lower_boot, comments='#', header='Lower limit for the Conditional distribution of {} given {} from bootstrap run'.format(X_label, Y_label))
        np.savetxt(os.path.join(aux_output_location,'X_cond_Y_upper_boot.txt'),X_cond_Y_upper_boot, comments='#', header='Upper limit for the Conditional distribution of {} given {} from bootstrap run'.format(X_label, Y_label))


@lru_cache(maxsize=200)
def _load_lookup_table(f_path):
    """
    Load the lookup table interpolate object and pass the object.
    INPUT:
        f_path : Entire file path for the .npy interpolated file.
    OUTPUT:
        lookup_inter : Interpolated lookup table (.npy) object.

    """

    lookup_inter = np.load(f_path, encoding = 'bytes', allow_pickle=True).item()
    print('Loaded lookup table from {}'.format(f_path))

    return lookup_inter


def _logging(message, filepath, verbose, append=True):
    """

    """

    message = str(current_process().pid)+":"+message

    if append:
        action="a"
    else:
        action="w"

    if verbose==1:
        with open(os.path.join(filepath,'log_file.txt'),action) as f:
            f.write('Using core '+message)
    elif verbose==2:
        with open(os.path.join(filepath,'log_file.txt'),action) as f:
            f.write('Using core '+message)
        print('Using core '+message)

    return 1

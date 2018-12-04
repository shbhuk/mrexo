import numpy as np
import os

from .mle_utils import cond_density_quantile

pwd = os.path.dirname(__file__)

def predict_m_given_r(Radius,  Radius_sigma = None, result_dir = None, dataset = 'mdwarf',
                      posterior_sample = False, qtl = [0.16,0.84], islog = False):
    '''
    Predict mass from given radius.
    Given radius can be a single measurement, with or without error, or can also be a posterior distribution of radii.

    INPUT:
        Radius: Numpy array of radius measurements.
        Radius_sigma: Numpy array of radius uncertainties. Assumes symmetrical uncertainty. Default : None

        result_dir: The directory where the results of the fit are stored. Default is None.
                If None, then will either use M-dwarf or Kepler fits (supplied with package).
        dataset: If result_dir == None, then will use included fits for M-dwarfs or Kepler dataset.
                To run the M-dwarf or Kepler set, define result_dir as None,
                and then dataset = 'mdwarf', or dataset = 'kepler'

                The Kepler dataset has been explained in Ning et al. 2018.
                The M-dwarf dataset has been explained in Kanodia et al. 2019.
        posterior_sample: If the input radii is a posterior sample, posterior_sample = True, else False.
                Default = False
        qtl = 2 element array or list with the quantile values that will be returned.
                Default is 0.16 and 0.84. qtl = [0.16,0.84]
        islog = Whether the radius given is in log scale or not.
                Default is False. The Radius_sigma is always in original units
    OUTPUT:
        outputs: Tuple with the predicted mass (or distribution of masses if input is a posterior),
                and the quantile distribution according to the 'qtl' input parameter

    EXAMPLE:
        #Below example predicts the mass for a radius of log10(1) Earth radii exoplanet, with no measurement uncertainty from the fit results in 'M_dwarfs_deg_cv'

        from mrexo import predict_m_given_r
        import os
        import numpy as np

        pwd = '~/mrexo_working/'
        result_dir = os.path.join(pwd,'M_dwarfs_deg_cv')

        predicted_mass, lower_qtl_mass, upper_qtl_mass = predict_m_given_r(Radius = 1, Radius_sigma = None, result_dir = result_dir, posterior_sample = False, islog = True)

        #Below example predicts the mass for a radius of log10(1) Earth radii exoplanet with uncertainty of 0.1 Earth Radii on the included Mdwarf fit. Similary for Kepler dataset.
        predicted_mass, lower_qtl_mass, upper_qtl_mass = predict_m_given_r(Radius = 1, Radius_sigma = 0.1, result_dir = None, dataset = 'mdwarf', posterior_sample = False, islog = True)

    '''

    # Define the result directory.
    mdwarf_resultdir = os.path.join(pwd, 'datasets', 'M_dwarfs_20181109')
    kepler_resultdir = os.path.join(pwd, 'datasets', 'Kepler_Ning_etal_20170605')

    if result_dir == None:
        if dataset == 'mdwarf':
            result_dir = mdwarf_resultdir
        elif dataset == 'kepler':
            result_dir = kepler_resultdir

    print(result_dir)

    input_location = os.path.join(result_dir, 'input')
    output_location = os.path.join(result_dir, 'output')

    # Load the results from the directory
    Mass_min, Mass_max = np.loadtxt(os.path.join(input_location, 'Mass_bounds.txt'))
    Radius_min, Radius_max = np.loadtxt(os.path.join(input_location, 'Radius_bounds.txt'))
    weights_mle = np.loadtxt(os.path.join(output_location,'weights.txt'))

    degrees = int(np.sqrt(len(weights_mle)))

    print(degrees)

    # Convert the radius measurement to log scale.
    if islog == False:
        logRadius = np.log10(Radius)
    else:
        logRadius = Radius

    # Check if single measurement or posterior distribution.
    if posterior_sample == False:
        predicted_value = cond_density_quantile(y = logRadius, y_std = Radius_sigma, y_max = Radius_max, y_min = Radius_min,
                                                      x_max = Mass_max, x_min = Mass_min, deg = degrees,
                                                      w_hat = weights_mle, qtl = qtl)
        predicted_mean = predicted_value[0]
        predicted_lower_quantile = predicted_value[2]
        predicted_upper_quantile = predicted_value[3]

        outputs = [predicted_mean,predicted_lower_quantile,predicted_upper_quantile]

    elif posterior_sample == True:

        n = np.size(Radius)
        mean_sample = np.zeros(n)
        random_quantile = np.zeros(n)

        if len(logRadius) != len(Radius_sigma):
            print('Length of Radius array is not equal to length of Radius_sigma array. CHECK!!!!!!!')
            return 0

        for i in range(0,n):
            qtl_check = np.random.random()
            print(qtl_check)
            results = cond_density_quantile(y = logRadius[i], y_std = Radius_sigma[i], y_max = Radius_max, y_min = Radius_min,
                                                      x_max = Mass_max, x_min = Mass_min, deg = degrees,
                                                      w_hat = weights_mle, qtl = [qtl_check,0.5])

            mean_sample[i] = results[0]
            random_quantile[i] = results[2]

        outputs = [mean_sample,random_quantile]

    if islog:
        return outputs
    else:
        return [10**x for x in outputs]



def predict_r_given_m(Mass,  Mass_sigma = None, result_dir = None, dataset = 'mdwarf',
                      posterior_sample = False, qtl = [0.16,0.84], islog = False):
    '''
    Predict radius from given mass.
    Given mass can be a single measurement, with or without error, or can also be a posterior distribution of mass.

    INPUT:
        Mass: Numpy array of mass measurements.
        Mass_sigma: Numpy array of mass uncertainties. Assumes symmetrical uncertainty. Default : None

        result_dir: The directory from where the results of the fit are read in. Default is None.
                If None, then will either use M-dwarf or Kepler fits (supplied with package).
        dataset: If result_dir == None, then will use included fits for M-dwarfs or Kepler dataset.
                To run the M-dwarf or Kepler set, define result_dir as None,
                and then dataset = 'mdwarf', or dataset = 'kepler'

                The Kepler dataset has been explained in Ning et al. 2018.
                The M-dwarf dataset has been explained in Kanodia et al. 2019.
        posterior_sample: If the input mass is a posterior sample, posterior_sample = True, else False.
                Default = False
        qtl = 2 element array or list with the quantile values that will be returned.
                Default is 0.16 and 0.84. qtl = [0.16,0.84]
        islog = Whether the radius given is in log scale or not.
                Default is False. The Radius_sigma is always in original units
    OUTPUT:
        outputs: Tuple with the predicted radius (or distribution of radii if input is a posterior),
                and the quantile distribution according to the 'qtl' input parameter

    EXAMPLE:
        #Below example predicts the radius for a mass of log10(1) Earth radii exoplanet, with no measurement uncertainty from the fit results in 'M_dwarfs_deg_cv'

        from mrexo import predict_r_given_m
        import os
        import numpy as np

        pwd = '~/mrexo_working/'
        result_dir = os.path.join(pwd,'M_dwarfs_deg_cv')

    '''

    # Define the result directory.
    mdwarf_resultdir = os.path.join(pwd, 'datasets', 'M_dwarfs_20181109')
    kepler_resultdir = os.path.join(pwd, 'datasets', 'M_dwarfs_20181109')

    if result_dir == None:
        if dataset == 'mdwarf':
            result_dir = mdwarf_resultdir
        elif dataset == 'kepler':
            result_dir = kepler_resultdir

    print(result_dir)

    input_location = os.path.join(result_dir, 'input')
    output_location = os.path.join(result_dir, 'output')

    # Load the results from the directory
    Mass_min, Mass_max = np.loadtxt(os.path.join(input_location, 'Mass_bounds.txt'))
    Radius_min, Radius_max = np.loadtxt(os.path.join(input_location, 'Radius_bounds.txt'))
    weights_mle = np.loadtxt(os.path.join(output_location,'weights.txt'))

    degrees = int(np.sqrt(len(weights_mle)))

    print(degrees)

    # Convert the mass measurement to log scale.
    if islog == False:
        logMass = np.log10(Mass)
    else:
        logMass = Mass

    # Check if single measurement or posterior distribution.
    if posterior_sample == False:
        predicted_value = cond_density_quantile(y = logMass, y_std = Mass_sigma, y_max = Mass_max, y_min = Mass_min,
                                                      x_max = Radius_max, x_min = Radius_min, deg = degrees,
                                                      w_hat = weights_mle, qtl = qtl)
        predicted_mean = predicted_value[0]
        predicted_lower_quantile = predicted_value[2]
        predicted_upper_quantile = predicted_value[3]

        outputs = [predicted_mean,predicted_lower_quantile,predicted_upper_quantile]

    elif posterior_sample == True:

        n = np.size(Mass)
        mean_sample = np.zeros(n)
        random_quantile = np.zeros(n)

        if len(logMass) != len(Mass_sigma):
            print('Length of Mass array is not equal to length of Mass_sigma array. CHECK!!!!!!!')
            return 0

        for i in range(0,n):
            qtl_check = np.random.random()
            print(qtl_check)
            results = cond_density_quantile(y = logMass[i], y_std = Mass_sigma[i], y_max = Mass_max, y_min = Mass_min,
                                                      x_max = Radius_max, x_min = Radius_min, deg = degrees,
                                                      w_hat = weights_mle, qtl = [qtl_check,0.5])

            mean_sample[i] = results[0]
            random_quantile[i] = results[2]

        outputs = [mean_sample,random_quantile]

    if islog:
        return outputs
    else:
        return [10**x for x in outputs]

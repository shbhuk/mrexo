import os
from astropy.table import Table
import numpy as np
from multiprocessing import cpu_count
import numpy as np


from mrexo import fit_xy_relation
from mrexo import predict_from_measurement, plot_joint_xy_distribution, plot_mle_weights, plot_yx_and_xy


try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''
    print('Could not find pwd')



'''
Sample script to fit mass-radius relationship.
The CSV table is generated from the NASA Exoplanet Archive. The existing example
is for the 24 M dwarf planets as explained in Kanodia 2019.
This can be replaced with any other CSV file.

For this sample, the cross validation has already been performed and the optimum number of
degrees has been established to be 17. For a new sample, set select_deg = 'cv' to
use cross validation to find the optimum number of degrees.

Can use parallel processing by setting cores > 1.
To use all the cores in the CPU, cores=cpu_count() (from multiprocessing import cpu_count)

To bootstrap and estimate the robustness of the median, set num_boot > 1.
If cores > 1, then uses parallel processing to run the various boots. For large datasets,
first run with num_boot to be a smaller number to estimate the computational time.

For more detailed guidelines read the docuemtnation for the fit_mr_relation() function.
'''


t = Table.read(os.path.join(pwd,'Cool_stars_20200520_exc_upperlim.csv'))
# t = Table.read(os.path.join(pwd,'Kepler_MR_inputs.csv'))


# Symmetrical errorbars
Mass_sigma = (abs(t['pl_masseerr1'])) #+ abs(t['pl_masseerr2']))/2
Radius_sigma = (abs(t['pl_radeerr1']))# + abs(t['pl_radeerr2']))/2

# In Earth units
Mass = np.array(t['pl_masse'])
Radius = np.array(t['pl_rade'])

# Directory to store results in
result_dir = os.path.join(pwd,'Mdwarfs_20200520_profile50')

# Run with 100 bootstraps. Selecting degrees to be 17. Alternatively can set select_deg = 'cv' to
# find the optimum number of degrees.

RadiusDict = {'X': Radius, 'X_sigma': Radius_sigma, 'X_max':None, 'X_min':None, 'X_label':'Radius', 'X_char':'r'}
MassDict = {'Y': Mass, 'Y_sigma': Mass_sigma, 'Y_max':None, 'Y_min':None, 'Y_label':'Mass', 'Y_char':'m'}

if __name__ == '__main__':
            initialfit_result, _ = fit_xy_relation(**RadiusDict, **MassDict,
                                                save_path = result_dir, select_deg = 'profile',
                                                num_boot = 5, cores = 2, degree_max=50)
"""
QueryRadii = [1, 3, 5, 8]
ExistingMR = np.zeros((len(QueryRadii), 3))
NewMR = np.zeros((len(QueryRadii), 3))

for i, r in enumerate(QueryRadii):
    ExistingMR[i, 0], predictionquantile, _ = predict_from_measurement(measurement=r, measurement_sigma=0.1*r)
    ExistingMR[i, 1:] = predictionquantile
    NewMR[i, 0], predictionquantile, _ = predict_from_measurement(measurement=r, measurement_sigma=0.1*r,
                result_dir=result_dir)
    NewMR[i, 1:] = predictionquantile

df = pd.DataFrame({"Radii":QueryRadii, "Old34planetdeg30":ExistingMR, "New34planet":NewMR)
df.to_csv(os.path.join(result_dir, 'output', 'other_data_products', 'PredictRadii.csv'), index=False)

fig, _ = plot_joint_xy_distribution(result_dir=result_dir)
fig.savefig(os.path.join(result_dir, 'output', 'other_data_products', 'JointDist.png')

fig, _ = plot_mle_weights(result_dir=result_dir)
fig.savefig(os.path.join(result_dir, 'output', 'other_data_products', 'Weights.png')

fig, _ = plot_yx_and_xy(result_dir=result_dir)
fig.savefig(os.path.join(result_dir, 'output', 'other_data_products', 'ConditionalDist.png')

"""


## **Sample script to predict Mass/Radius**

Sample script to show how to use the predicting function to predict mass from radius

```python
from mrexo import predict_from_measurement, generate_lookup_table
import os
import numpy as np
import matplotlib.pyplot as plt


try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''
    print('Could not find pwd')

```

Sample script to show how to use the predicting function to predict mass from radius



**I.** Predict mass and quantiles from radius for a 1 Earth radii planet with an uncertainty of 0.1 radii using the M dwarf fit from the result_dir
```python
result_dir = os.path.join(pwd,'M_dwarfs_dummy')
predicted_mass, qtls, iron_planet = predict_from_measurement(measurement=1, measurement_sigma=0.1, result_dir=result_dir, is_posterior=False, show_plot=False)
print(predicted_mass)
```
**II.** Predict mass from radius for the Kepler dataset for a 1 Earth radii planet
```python
predicted_mass, qtls, iron_planet = predict_from_measurement(measurement=1, measurement_sigma=None, predict = 'mass', dataset='kepler')
```

**III.** Predict the mass measurement from a dummy radius posterior and plot it
```python
posterior, iron_planet = predict_from_measurement(measurement=np.random.normal(1,0.1,1000),
            measurement_sigma=None, result_dir=None, dataset='mdwarf', is_posterior=True, show_plot=True, use_lookup = True)
```

**IV.** Predict the mass for a radius of 1 Earth radii exoplanet with uncertainty of 0.1 Earth Radii on the included Mdwarf fit.
Also output 5,16,84,95% quantile

```python
predicted_mass, qtls, iron_planet = predict_from_measurement(measurement=1, measurement_sigma=0.1, result_dir=None, dataset='mdwarf', is_posterior=False,
                       qtl = [0.05,0.16,0.84,0.95], show_plot=False)
```

Alternatively, the default dataset is the M dwarf dataset from Kanodia 2019.                       

```python 
predicted_mass, qtls, iron_planet = predict_from_measurement(measurement=1, measurement_sigma=0.1,qtl = [0.05,0.16,0.84,0.95])
```

==================



## **Sample script to plot Mass-Radius relationships**

Sample script to show how to use the plotting functions available with MRExo

```python
from mrexo import plot_r_given_m_relation, plot_m_given_r_relation, plot_mr_and_rm, plot_joint_mr_distribution
import os
import numpy as np
import matplotlib.pyplot as plt


try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''
    print('Could not find pwd')

```

Sample script to show how to use the plotting functions for the M dwarf dataset from Kanodia (2019)

```python
mdwarf_result = r'C:\Users\shbhu\Documents\GitHub\mrexo\mrexo\datasets\M_dwarfs_20181214'
kepler_result = r'C:\Users\shbhu\Documents\Git\mrexo\mrexo\datasets\Kepler_Ning_etal_20170605'

result_dir = mdwarf_result
```

**I.** Plot the conditional distribution f(m|r)
```python
ax = plot_m_given_r_relation(result_dir)
```

** II.** Plot the conditional distribution f(r|m)
```python
ax = plot_r_given_m_relation(result_dir)
```

** III.** Plot both the conditional distributions f(m|r) and f(r|m), similar to Kanodia 2019, Fig 3.
```python
ax = plot_mr_and_rm(result_dir)
```

** IV.** Plot the joint distribution f(m,r)
```python
ax = plot_joint_mr_distribution(result_dir)
```


==================



## **Sample script to fit mass-radius relationship**

The CSV table is generated from the NASA Exoplanet Archive. The existing example
is for the 24 M dwarf planets as explained in Kanodia 2019.
This can be replaced with a different dataset CSV file.

For this sample, the cross validation has already been performed and the optimum number of 
degrees has been established to be 17. For a new sample, set select_deg = 'cv' to 
use cross validation to find the optimum number of degrees. 

Can use parallel processing by setting cores > 1. 
To use all the cores in the CPU, cores=cpu_count() (from multiprocessing import cpu_count)

To bootstrap and estimate the robustness of the median, set num_boot > 1. 
If cores > 1, then uses parallel processing to run the various boots. For large datasets,
first run with num_boot to be a smaller number to estimate the computational time.

For more detailed guidelines read the docuemtnation for the fit_mr_relation() function. 

```python
import os
from astropy.table import Table
import numpy as np
from multiprocessing import cpu_count
import numpy as np


from mrexo import fit_mr_relation


try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''
    print('Could not find pwd')

t = Table.read(os.path.join(pwd,'Cool_stars_MR_20181214_exc_upperlim.csv'))

# Symmetrical errorbars
Mass_sigma = (abs(t['pl_masseerr1']) + abs(t['pl_masseerr2']))/2
Radius_sigma = (abs(t['pl_radeerr1']) + abs(t['pl_radeerr2']))/2

# In Earth units
Mass = np.array(t['pl_masse'])
Radius = np.array(t['pl_rade'])

# Directory to store results in
result_dir = os.path.join(pwd,'M_dwarfs_new_cv')

# Run with 50 bootstraps. Selecting degrees to be 17. Alternatively can set select_deg = 'cv' to
# find the optimum number of degrees.

if __name__ == '__main__':
            initialfit_result, bootstrap_results = fit_mr_relation(Mass = Mass, Mass_sigma = Mass_sigma,
                                                Radius = Radius, Radius_sigma = Radius_sigma,
                                                save_path = result_dir, select_deg = 17,
                                                num_boot = 50, cores = cpu_count())
```

==================



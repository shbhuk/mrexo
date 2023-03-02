import numpy as np
import pandas as pd



# To read the stellar properties table:
table_stars_colnames = ['KOI', 'Gaia', 'Ksmag', 'e_Ksmag', 'par', 'e_par', 'Teff', 'e_Teff', 'FeH', 'e_FeH', 'vsini', 'Rstar', 'E_Rstar', 'e_Rstar', 'Mstar-iso', 'E_Mstar-iso', 'e_Mstar-iso', 'Rstar-iso', 'E_Rstar-iso', 'e_Rstar-iso', 'rho-iso', 'E_rho-iso', 'e_rho-iso', 'age-iso', 'E_age-iso', 'e_age-iso', 'par-spec', 'E_par-spec', 'e_par-spec', 'prov', 'SB2', 'CXM']
#table_stars_dtypes = {'KOI':int, 'Gaia':int, 'Ksmag':float, 'e_Ksmag':float, 'par':float, 'e_par':float, 'Teff':int, 'e_Teff':int, 'FeH':float, 'e_FeH':float, 'vsini':float, 'Rstar':float, 'E_Rstar':float, 'e_Rstar':float, 'Mstar-iso':float, 'E_Mstar-iso':float, 'e_Mstar-iso':float, 'Rstar-iso':float, 'E_Rstar-iso':float, 'e_Rstar-iso':float, 'rho-iso':float, 'E_rho-iso':float, 'e_rho-iso':float, 'age-iso':float, 'E_age-iso':float, 'e_age-iso':float, 'par-spec':float, 'E_par-spec':float, 'e_par-spec':float, 'prov':str, 'SB2':int, 'CXM':int}
#table_stars_colwidths = [4, 19, 5, 4, 5, 4, 4, 3, 5, 4, 4, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 3, 4, 5, 4, 7, 3, 1, 1] # didn't work (fails after the 2nd column) for some reason

#table_stars = pd.read_csv('CKS-X_stellar_params.txt', names=table_stars_colnames, skiprows=76, delim_whitespace=True)
table_stars = pd.read_fwf('CKS-X_stellar_params.txt', infer_nrows=500, names=table_stars_colnames, skiprows=76, skipfooter=4, delim_whitespace=True) # close enough; for 'SB2' column, values of 'syn' are read as 'yn' instead; also skips the last four rows which have no KOI numbers



# To read the planet properties table:
table_planets_colnames = ['Planet', 'Per', 'E_Per', 'e_Per', 'Rp/Rstar', 'E_Rp/Rstar', 'e_Rp/Rstar', 'T', 'E_T', 'e_T', 'Rp', 'E_Rp', 'e_Rp', 'Tmax-c', 'E_Tmax-c', 'e_Tmax-c', 'a', 'E_a', 'e_a', 'S', 'E_S', 'e_S', 'Samp']

table_planets = pd.read_fwf('CKS-X_planet_params.txt', infer_nrows=500, names=table_planets_colnames, skiprows=39, delim_whitespace=True)



# To combine the stellar and planet properties into one table (i.e. append the relevant stellar properties to the planets they host):
stars_colnames_keep = ['KOI', 'Teff', 'e_Teff', 'FeH', 'e_FeH', 'Rstar', 'E_Rstar', 'e_Rstar', 'Mstar-iso', 'E_Mstar-iso', 'e_Mstar-iso', 'Rstar-iso', 'E_Rstar-iso', 'e_Rstar-iso', 'age-iso', 'E_age-iso', 'e_age-iso']
table_stars_colskeep = table_stars[stars_colnames_keep]

data_combined = pd.DataFrame()
for i in range(len(table_planets)):
    tp_row = table_planets.iloc[i]
    koi_pl = table_planets['Planet'].iloc[i]
    koi = int(koi_pl[1:6])
    ts_dfrow = table_stars_colskeep[table_stars_colskeep['KOI'] == koi]
    assert len(ts_dfrow) == 1 # check that one and only one star matches
    ts_row = ts_dfrow.iloc[0]
    
    tps_row = pd.concat([tp_row, ts_row])
    data_combined = pd.concat([data_combined, tps_row.to_frame().T], ignore_index=True)

data_combined.to_csv('CKS-X_planets_stars.csv') # saves the new DataFrame to a CSV file!

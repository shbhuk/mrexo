from mrexo.predict import generate_lookup_table

#Sample script to generate look up table for predicting mass given radius. 
#To predict radius given mass, predict_quantity = 'mass'


kepler_result = '/storage/home/s/szk381/work/mrexo/mrexo/datasets/Kepler_Ning_etal_20170605'


generate_lookup_table(result_dir = kepler_result, predict_quantity = 'Mass')

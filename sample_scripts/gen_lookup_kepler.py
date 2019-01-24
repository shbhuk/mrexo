from mrexo.predict import generate_lookup_table
result_dir = '/storage/home/s/szk381/work/mrexo/mrexo/datasets/Kepler_Ning_etal_20170605'
kepler_result = r'C:\Users\shbhu\Documents\GitHub\mrexo\mrexo\datasets\Kepler_Ning_etal_20170605'

generate_lookup_table(result_dir = kepler_result, predict_quantity = 'Radius')

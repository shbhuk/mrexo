from multiprocessing import Pool
import numpy as np
 
def doubler(dummy):
    return np.random.uniform()
 

no_boots = 10

if __name__ == '__main__':
    numbers = np.arange(1,no_boots)
    pool = Pool(processes=3)
    print(pool.map(doubler,numbers))
a = 3
a_std = 1
a_LSigma = 1
a_USigma = 1

start = datetime.datetime.now()
for i in range(1000):
    _ComputeOldConvolvedPDF(a, deg, deg_vec, a_max, a_min, a_std=a_std, abs_tol=1e-8, Log=False)
end = datetime.datetime.now()
print(end-start)

start = datetime.datetime.now()
for i in range(1000):
    _ComputeConvolvedPDF(a, deg, deg_vec, a_max, a_min, 
        a_LSigma=a_LSigma, a_USigma=a_USigma,
        abs_tol=1e-8, Log=False)
end = datetime.datetime.now()
print(end-start)

i = 50
dim = 1

a=DataDict["ndim_data"][dim][i]
a_LSigma=DataDict["ndim_LSigma"][dim][i]
a_USigma=DataDict["ndim_USigma"][dim][i]
deg=deg_per_dim[dim]
deg_vec=deg_vec_per_dim[dim]
a_max=DataDict["ndim_bounds"][dim][1]
a_min=DataDict["ndim_bounds"][dim][0]

"""
start = datetime.datetime.now()
for i in range(10000):
	_ = stats.norm.pdf(i, i, 100)
end = datetime.datetime.now()
print(end-start)

start = datetime.datetime.now()
for i in range(1000000):
	_ = _PDF_Normal(i, i, 100)
end = datetime.datetime.now()
print(end-start)

n = 103

start = datetime.datetime.now()
for i in range(1000000):
	_ = _GammaFunction(n)
end = datetime.datetime.now()
print(end-start)

from math import factorial, gamma

start = datetime.datetime.now()
for i in range(1000000):
	_ = factorial(n-1)
end = datetime.datetime.now()
print(end-start)

start = datetime.datetime.now()
for i in range(1000000):
	_ = scipy.math.factorial(n-1)
end = datetime.datetime.now()
print(end-start)
"""

R_points = np.array(pd.read_csv(r"C:\Users\skanodia\Downloads\Compare_MRExo\Test_N100_Diagonal_0.1Error_Scatter\R_points.csv").iloc[:,1])
M_points = np.array(pd.read_csv(r"C:\Users\skanodia\Downloads\Compare_MRExo\Test_N100_Diagonal_0.1Error_Scatter\M_points.csv").iloc[:,1])

qtls = np.arange(0, 100)

# ECDF for mass given R=6 
M_cond_R6 = pd.read_csv(r"C:\Users\skanodia\Downloads\Compare_MRExo\Test_N100_Diagonal_0.1Error_Scatter\M_cond_R6_qtl100.csv")
R6_cdf = np.array(M_cond_R6.iloc[:,1])

InterpCDF = interp1d(qtls, R6_cdf, bounds_error=False, fill_value=(R6_cdf[0], R6_cdf[-1]))

RSample = []

for i in range(100000):
	p = np.random.uniform(0, 1)*100
	RSample.append(InterpCDF(p))

plt.hist(RSample, density=True)

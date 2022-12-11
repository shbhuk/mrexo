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


start = datetime.datetime.now()
for i in range(1000000):
	_ = _GammaFunction(323)
end = datetime.datetime.now()
print(end-start)

start = datetime.datetime.now()
for i in range(1000000):
	_ = Factorial(323)
end = datetime.datetime.now()
print(end-start)

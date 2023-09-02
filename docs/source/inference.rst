.. _inference:

⚡️ Inference ⚡️
=================================

After performing a joint fit following the example :ref:`here <fitting>`, ``MRExo`` can be used to condition the joint probability distribution (PDF) to predict one set of variable/s from the others. 
The user can also condition the  Joint PDFs from the Monte-Carlo or Bootstrap simulations to obtain posteriors quantifying the impact of measurement uncertainties or finite sample size (see `Kanodia et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023arXiv230810615K>`_  ).

2D Distributions
--------------------
For 2D fits, say --- *f(x, y)* ---,  the 2D PDF can be conditioned on a given measurement Y,  *y=Y*, to obtain the 1D PDF --- *f(x|y=Y)*, from which the user can obtain the expectation and variance of the distribution.   
Here *x, y* can refer to any two measured quantities. The sample script for this is included `here <https://github.com/shbhuk/mrexo/blob/master/sample_scripts/2D_marginalize1Dplot.py>`_  . 

3D Distributions conditioned on 1D 
--------------------
For 3D fits, say --- *f(x, y, z)* ---,  the 3D PDF can be conditioned on a given measurement Z,  *z=Z*, to obtain the  1D PDF --- *f(x, y|z=Z)*, from which the user can obtain the expectation and variance of the distribution.   
Here *x, y, z* can refer to any three measured quantities. The sample script for this is included `here <https://github.com/shbhuk/mrexo/blob/master/sample_scripts/3D_marginalize1Dplot.py>`_  . 

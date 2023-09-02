.. _inference:

⚡️ Inference ⚡️
=================================

After performing a joint fit following the example :ref:`here <fitting>`, ``MRExo`` can be used to condition the joint probability distribution (PDF) to predict one set of variable/s from the others. 

2D Inference
--------------------
For 2D fits, say --- 2D *f(m, r)* ---,  the 2D PDF can be conditioned on a given radius *r=Rp*, to obtain the PDF --- *f(m|r=Rp)*, from which the user can obtain the expectation and variance of the distribution. 


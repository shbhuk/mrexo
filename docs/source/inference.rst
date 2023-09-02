.. _inference:

⚡️ Inference ⚡️
=================================

After performing a joint fit following the example :ref:`here <fitting>`, ``MRExo`` can be used to condition the joint probability distribution (PDF) to predict one set of variable/s from the others. 
For example, following a 2D *f(m, r)* mass-radius fit, the 2D PDF can be conditioned on a given radius *r=Rp*, to obtain the PDF --- *f(m|r=Rp)*, from which one can obtain the expectation and variance of the distribution.

An example of how to load a sample and perform a fit in three dimensions is included in  `sample_fit.py <https://github.com/shbhuk/mrexo/blob/master/sample_scripts/sample_fit.py>`_ , where the same script can be adapted to fit two or four dimensions.

.. literalinclude:: ../../sample_scripts/sample_fit.py
   :language: python
   :emphasize-lines: 33,97,98,109
   :linenos:


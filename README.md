# MRExo 
Non parametric exoplanet mass radius relationship

We translate Ning et al. (2018)’s `R` script\[1\] into a publicly available `Python`
package called `MRExo`\[2\]. 

`MRExo` offers tools for fitting the M-R relationship to a given data
set.  Along with the `MRExo` installation, the fit
results from the M dwarf sample dataset from Kanodia et al. (2019) and the Kepler
exoplanet sample from  Ning et al. (2018) are included. The code also includes predicting functions (M->R, and R->M), and plotting functions to generate the plots used in the below preprint.

For detailed description of the code please see our preprint manuscript - 
http://adsabs.harvard.edu/abs/2019arXiv190300042K

We are waiting for the referee review, and are also working on a detailed tutorial for using this package. After that will also upload the package on to PyPI for easy `pip` install. In the meanwhile, feel free to email me or raise an issue if you have any issues with running this code. 

Steps to install barycorrpy from github - 

### Steps to Install from GitHub 
Linux/Unix systems - 


1. In the terminal - 
 `pip install git+https://github.com/shbhuk/mrexo.git -U`

OR
1. In the terminal - 
`pip install git+ssh://git@github.com/shbhuk/mrexo.git -U `

OR 

1. In the terminal - 
`git clone https://github.com/shbhuk/mrexo`

2. In the repository directory 
`python setup.py install`

The -U is to upgrade the dependencies.


Windows - 
 1. Install Git - [Git for Windows](https://git-for-windows.github.io/)
 2. Install pip - [pip for Windows](https://pip.pypa.io/en/stable/installing/)
 3. In cmd or git cmd - 
  `pip install git+https://github.com/shbhuk/mrexo.git -U`



1.  <https://github.com/Bo-Ning/Predicting-exoplanet-mass-and-radius-relationship>.

2.  <https://github.com/shbhuk/mrexo>

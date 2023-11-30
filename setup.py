from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()



setup(name='mrexo',
      version='1.1.3',
      description='Nonparametric mass radius+ inference for exoplanets',
      long_description=readme(),
      long_description_content_type="text/markdown",
      url='https://github.com/shbhuk/mrexo',
      author='Shubham Kanodia',
      author_email='shbhuk@gmail.com',
      install_requires=['astropy>2','matplotlib','numpy','scipy>1.8', 'pandas'],
      packages=['mrexo'],
      include_package_data = True,
      license='GPLv3',
      classifiers=['Topic :: Scientific/Engineering :: Astronomy'],
      keywords='Mass-Radius+ relationship Non parametric Exoplanets' )

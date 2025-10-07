#!/usr/bin/env python

# Read module version from init file
with open('dwdatareader/__init__.py') as f:
    for line in f:
        if line.startswith('__version__'):
            exec(line)
            break

from distutils.core import setup
setup(name='dwdatareader',
      version=__version__,
      description='Python module to interact with Dewesoft DWDataReaderLib shared library',
      long_description=open('README.rst').read(),
      author='Carl Osterwisch',
      author_email='costerwi@gmail.com',
      url='https://github.com/costerwi/dwdatareader/',
      download_url='https://github.com/costerwi/dwdatareader/tarball/master',
      license='MIT',
      packages=['dwdatareader'],
      package_data={'dwdatareader': ['DW*.dll', 'DW*.so']},
      classifiers = [
         "Intended Audience :: Science/Research",
         "Topic :: Scientific/Engineering :: Information Analysis",
         "Programming Language :: Python :: 2.7",
         "Programming Language :: Python :: 3",
         "Operating System :: Microsoft :: Windows",
         "Operating System :: POSIX :: Linux",
         "Development Status :: 4 - Beta",
         ],
     )

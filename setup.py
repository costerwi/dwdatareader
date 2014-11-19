#!/usr/bin/env python

from distutils.core import setup

setup(name='dwdatareader',
      version='0.1.2',
      description='Python module to interact with Dewesoft DWDataReaderLib shared library',
      long_description=open('README.md').read(),
      author='Carl Osterwisch',
      author_email='costerwi@gmail.com',
      url='https://github.com/costerwi/dwdatareader/',
      license='MIT',
      packages=[''],  # This generates a warning: invalid package name
      package_dir={'': 'dwdatareader'},
      package_data={'': ['*.dll']},
     )

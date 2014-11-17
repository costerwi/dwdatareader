#!/usr/bin/env python

from distutils.core import setup

setup(name='dwdatareader',
      version='0.1.1',
      description='Python module to interact with Dewesoft DWDataReaderLib shared library',
      author='Carl Osterwisch',
      author_email='costerwi@gmail.com',
      url='https://github.com/costerwi/dwdatareader/',
      packages=['dwdatareader'],
      package_data={'dwdatareader': ['*.dll']},
     )

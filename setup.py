#!/usr/bin/env python

from distutils.core import setup
exec(open('dwdatareader/_version.py'))
setup(name='dwdatareader',
      version=__version__,
      description='Python module to interact with Dewesoft DWDataReaderLib shared library',
      long_description=open('README.md').read(),
      author='Carl Osterwisch',
      author_email='costerwi@gmail.com',
      url='https://github.com/costerwi/dwdatareader/',
      license='MIT',
      packages=['dwdatareader'],
      package_data={'dwdatareader': ['*.dll']},
     )

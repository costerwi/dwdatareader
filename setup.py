#!/usr/bin/env python

# Read module version from init file
with open('dwdatareader/__init__.py') as f:
    for line in f:
        if line.startswith('__version__'):
            exec(line)

from distutils.core import setup
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

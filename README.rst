dwdatareader
============

.. image:: https://travis-ci.com/costerwi/dwdatareader.svg?branch=master
   :alt: DWDataReader build status on Travis CI
   :target: https://travis-ci.com/costerwi/dwdatareader

.. image:: https://ci.appveyor.com/api/projects/status/a2qssrmuepbx224i/branch/master?svg=true
   :alt: DWDataReader build status on Appveyor
   :target: https://ci.appveyor.com/project/costerwi/dwdatareader/branch/master

DEWESoft produces hardware and software for test measurement, data aquisition, 
and storage. Data files are stored with the extension .d7d or .dxd in a proprietary
format. DEWESoft provides a free Windows application (DewesoftX) to work with the data
and a free shared library for developers.

This is a Python module to interact with the DEWESoft DWDataReaderLib shared library, which can be downloaded from https://dewesoft.com/download/developer-downloads.

Installation
------------

The module is available on https://pypi.python.org/pypi/dwdatareader so all
one needs to do is:

::

    pip install dwdatareader

Example usage
-------------

Scripts like the following may be run from the command line or, more
interactively, from `Jupyter Notebook <http://jupyter.org>`_

You can work with a `live Binder example here <https://mybinder.org/v2/gh/costerwi/dwdatareader/master?labpath=dwdatareader_example.ipynb>`_.

.. code:: python

    import dwdatareader as dw
    with dw.open('myfile.d7d') as f:
        print(f.info)
        ch1 = f['chname1'].series()
        ch1.plot()
        for ch in f.values():
            print(ch.name, ch.series().mean())


Contribute
----------

Bug reports and pull requests should be directed to the project home on
`Github <http://github.com/costerwi/dwdatareader>`_

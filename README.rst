dwdatareader
============

DEWESoft produces hardware and software for test measurement, data aquisition, 
and storage. Data files are stored with the extension .d7d in a proprietary
format. DEWESoft provides a free Windows application to work with the data
and a free shared library for developers on Windows and Linux.

This is a Python module to interact with the DEWESoft DWDataReaderLib shared library
available from http://www.dewesoft.com/developers

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

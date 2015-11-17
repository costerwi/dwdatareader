dwdatareader
============

Fork from https://github.com/costerwi/dwdatareader
Change to newer version of the DWDataReaderLib *DWDataReader_v4_1_0_12.zip* and correct some function calls.

Python module to interact with Dewesoft DWDataReaderLib shared library
available from http://www.dewesoft.com/developers

Installation
------------
::

    pip install http://github.com/costerwi/dwdatareader/tarball/master

Example usage
-------------
.. code:: python

    from dwdatareader import dwdatareader as dw
    with dw.open('myfile.d7d') as f:
        print(f.info)
        ch1 = f['chname1'].series()
        ch1.plot()
        for ch in f.values():
            print(ch.name, ch.series().mean())

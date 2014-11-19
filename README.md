dwdatareader
============

Python module to interact with Dewesoft DWDataReaderLib shared library
available from http://www.dewesoft.com/developers

Installation:
```
pip install http://github.com/costerwi/dwdatareader/tarball/master
```

Example usage:
```python
import dwdatareader
with dwdatareader.DW() as dw:
    with dw.open('myfile.d7d') as f:
        print f.info
        ch1 = f['chname1'].series()
        ch1.plot()
        for ch in f.values():
            print ch.name, ch.series().mean()
```

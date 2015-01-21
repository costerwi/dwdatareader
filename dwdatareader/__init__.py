"""Python module to interact with Dewesoft DWDataReaderLib.dll

@author: shelmore and costerwisch

Example usage:
import dwdatareader
with dwdatareader.DW() as dw:
    with dw.open('myfile.d7d') as f:
        print f.info
        ch1 = f['chname1'].series()
        ch1.plot()
        for ch in f.values():
            print ch.name, ch.series().mean()
"""

__all__ = ['DW', 'DWError']
__version__ = '0.3.2'
from interface import DW, DWError

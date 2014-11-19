# -*- coding: utf-8 -*-
"""
Classes to interface with Dewesoft DWDataReaderLib shared library

@author: shelmore and costerwisch
"""

import collections
import ctypes
from exceptions import RuntimeError, KeyError

class DWError(RuntimeError):
    """Interpret error number returned from dll"""
    errors = ["status OK", "error in DLL", "cannot open d7d file",
            "file already in use", "d7d file corrupt", "memory allocation"]
            
    def __init__(self, value):
        RuntimeError.__init__(self, self.errors[value])


class DWInfo(ctypes.Structure):
    """Structure to hold metadata about DWFile"""
    _fields_ = [("sample_rate", ctypes.c_double),
                ("start_store_time", ctypes.c_double),
                ("duration", ctypes.c_double)]

    def __str__(self):
        return "{0.start_store_time} {0.sample_rate} Hz {0.duration} s".format(self)
        
    def datetime(self):
        """Return start_store_time in Python datetime format"""
        import datetime
        import pytz
        epoch = datetime.datetime(1899, 12, 30, tzinfo=pytz.utc)
        return epoch + datetime.timedelta(self.start_store_time)


class DWReducedValue(ctypes.Structure):
    _fields_ = [("time_stamp", ctypes.c_double),
                ("ave", ctypes.c_double),
                ("min", ctypes.c_double),
                ("max", ctypes.c_double),
                ("rms", ctypes.c_double)]
                
    def __str__(self):
        return "{0.time_stamp} {0.ave} ave".format(self)

        
class DWChannel(ctypes.Structure):
    """Store channel metadata, provide methods to load channel data"""
    _fields_ = [("index", ctypes.c_int),
                ("name", ctypes.c_char * 100),
                ("unit", ctypes.c_char * 20),
                ("description", ctypes.c_char * 200),
                ("color" , ctypes.c_uint),
                ("array_size", ctypes.c_int)]
                
    def __str__(self):
        return "{0.name} ({0.unit}) {0.description}".format(self)
        
    def scaled(self):
        """Load full speed data"""
        import numpy
        count = DW.DLL.DWGetScaledSamplesCount(self.index)
        data = numpy.empty(count, dtype=numpy.double)
        time = numpy.empty_like(data)
        stat = DW.DLL.DWGetScaledSamples(self.index, 0, count,
                data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                time.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if stat:
            raise DWError(stat)
        return time, data
        
    def reduced(self):
        """Load reduced (averaged) data"""
        count = ctypes.c_int()
        block_size = ctypes.c_double()
        stat = DW.DLL.DWGetReducedValuesCount(self.index, 
            ctypes.pointer(count), ctypes.pointer(block_size))
        if stat:
            raise DWError(stat)
        data = (DWReducedValue * count.value)()
        stat = DW.DLL.DWGetReducedValues(self.index, 0, count,
                ctypes.pointer(data))
        if stat:
            raise DWError(stat)
        return data
        
    def series(self):
        """Load and return timeseries of results for channel"""
        import pandas
        time, data = self.scaled()
        if not len(data):
            # Use reduced data if scaled is not available
            r = self.reduced()
            time = [i.time_stamp for i in r]
            data = [i.ave for i in r]
        return pandas.Series(data = data, index = time, name = self.name)


class DWFile(collections.Mapping):
    """Data file type mapping channel names their metadata"""

    def __init__(self, name):
        self.name = name  # filename to open
        
        # Open the file
        self.info = DWInfo()
        stat = DW.DLL.DWOpenDataFile(name.encode(), self.info)
        if stat:
            raise DWError(stat)

        # Read file header section
        self.header = dict()
        name = ctypes.create_string_buffer(100)
        text = ctypes.create_string_buffer(200)
        nHeaders = DW.DLL.DWGetHeaderEntryCount()
        for i in range(nHeaders):
            stat = DW.DLL.DWGetHeaderEntryTextF(i, text, len(text))
            if stat:
                raise DWError(stat)
            if len(text.value) and not(text.value.startswith('Select...') or 
                    text.value.startswith('To fill out')):
                stat = DW.DLL.DWGetHeaderEntryNameF(i, name, len(name))
                if stat:
                    raise DWError(stat)
                self.header[name.value] = text.value
            
        # Read channel metadata
        nchannels = DW.DLL.DWGetChannelListCount()
        self.channels = (DWChannel * nchannels)()
        stat = DW.DLL.DWGetChannelList(self.channels)
        if stat:
            raise DWError(stat)
            
    def close(self):
        DW.DLL.DWCloseDataFile()
        self.channels = [] # prevent access to closed file
        
    def __len__(self):
        return len(self.channels)

    def __getitem__(self, key):
        for ch in self.channels: # brute force lookup
            if ch.name == key or ch.index == key:
                return ch
        raise KeyError(key)
                
    def __iter__(self):
        for ch in self.channels:
            yield ch.name
            
    def __str__(self):
        return self.name
        
    def __enter__(self):
        """Used to maintain file in context"""
        return self

    def __exit__(self, type, value, traceback):
        """Close file when it goes out of context"""
        self.close()


class DW:
    """Initialize and de-initialize the Dewesoft data reader dll library"""
    DLL = None # class variable accessible to other classes
    
    def __init__(self, dllName = None):
        """Load the dll library"""
        # Dynamic library available from http://www.dewesoft.com/developers
        import os
        if not dllName:
            dllName = os.path.join(os.path.dirname(__file__),
                "DWDataReaderLib64")
        self.name = dllName
        DW.DLL = ctypes.cdll.LoadLibrary(dllName)

    def open(self, name):
        """Open the specified data file"""
        return DWFile(name)

    def close(self):
        """Close the data file"""
        self.DLL.DWCloseDataFile()

    def init(self):
        """Initialize the dll"""
        stat = self.DLL.DWInit()
        if stat:
            raise DWError(stat)
        
    def deInit(self):
        """Shutdown the dll"""
        self.close() # Required to prevent crash if file is open
        self.DLL.DWDeInit()
    
    def __str__(self):
        return "{} {}".format(self.name, self.DLL.DWGetVersion())
    
    def __enter__(self):
        """Used to maintain library in context"""
        self.Init()
        return self
        
    def __exit__(self, type, value, traceback):
        """Close library when it goes out of context"""
        self.deInit()

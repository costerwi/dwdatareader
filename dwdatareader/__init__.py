"""Python module to interact with Dewesoft DWDataReaderLib.dll

@author: shelmore and costerwisch

Example usage:
import dwdatareader as dw
with dw.open('myfile.d7d') as f:
    print f.info
    ch1 = f['chname1'].series()
    ch1.plot()
    for ch in f.values():
        print ch.name, ch.series().mean()
"""
__all__ = ['DWError', 'DWFile']
__version__ = '0.7.0'

DLL = None # module variable accessible to other classes 

import collections
import ctypes

class DWError(RuntimeError):
    """Interpret error number returned from dll"""
    errors = ["status OK", "error in DLL", "cannot open d7d file",
            "file already in use", "d7d file corrupt", "memory allocation"]
            
    def __init__(self, value):
        super(DWError, self).__init__(self.errors[value])


class DWInfo(ctypes.Structure):
    """Structure to hold metadata about DWFile"""
    _pack_ = 1
    _fields_ = [("sample_rate", ctypes.c_double),
                ("_start_store_time", ctypes.c_double),
                ("duration", ctypes.c_double)]

    def __str__(self):
        return "{0.start_store_time} {0.sample_rate} Hz {0.duration} s".format(self)

    @property
    def start_store_time(self):
        """Return start_store_time in Python datetime format"""
        import datetime
        import pytz
        epoch = datetime.datetime(1899, 12, 30, tzinfo=pytz.utc)
        return epoch + datetime.timedelta(self._start_store_time)


class DWReducedValue(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("time_stamp", ctypes.c_double),
                ("ave", ctypes.c_double),
                ("min", ctypes.c_double),
                ("max", ctypes.c_double),
                ("rms", ctypes.c_double)]
                
    def __str__(self):
        return "{0.time_stamp} {0.ave} ave".format(self)


class DWEvent(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("event_type", ctypes.c_int),
                ("time_stamp", ctypes.c_double),
                ("_event_text", ctypes.c_char * 200)]

    @property
    def event_text(self):
        return self._event_text.decode()

    def __str__(self):
        return "{0.time_stamp} {0.event_text}".format(self)
        
    
class DWChannel(ctypes.Structure):
    """Store channel metadata, provide methods to load channel data"""
    _pack_ = 1
    _fields_ = [("index", ctypes.c_int),
                ("_name", ctypes.c_char * 100),
                ("_unit", ctypes.c_char * 20),
                ("_description", ctypes.c_char * 200),
                ("color" , ctypes.c_uint),
                ("array_size", ctypes.c_int)]
    
    @property
    def name(self):
        return self._name.decode()

    @property
    def unit(self):
        return self._unit.decode()

    @property
    def description(self):
        return self._description.decode()

    def __str__(self):
        return "{0.name} ({0.unit}) {0.description}".format(self)

    def scaled(self):
        """Load full speed data"""
        import numpy
        count = DLL.DWGetScaledSamplesCount(self.index)
        data = numpy.empty(count, dtype=numpy.double)
        time = numpy.empty_like(data)
        stat = DLL.DWGetScaledSamples(self.index, 0, count,
                data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                time.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if stat:
            raise DWError(stat)
        return time, data
        
    def reduced(self):
        """Load reduced (averaged) data"""
        count = ctypes.c_int()
        block_size = ctypes.c_double()
        stat = DLL.DWGetReducedValuesCount(self.index, 
            ctypes.pointer(count), ctypes.pointer(block_size))
        if stat:
            raise DWError(stat)
        data = (DWReducedValue * count.value)()
        stat = DLL.DWGetReducedValues(self.index, 0, count,
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
        return pandas.Series(data = data, index = time, 
                             name = self.name)

    def plot(self, *args, **kwargs):
        """Plot the data as a series"""
        ax = self.series().plot(*args, **kwargs)
        ax.set_ylabel(self.unit)
        return ax


class DWFile(collections.Mapping):
    """Data file type mapping channel names their metadata"""

    closed = True  # bool indicating the current state of the file object  
    
    def __init__(self, name = None):
        if name:
            self.open(name)

    def open(self, name = None):
        """Open the specified file and read channel headers"""
        
        if not name:
            name = self.name # reopen previous file
        info = DWInfo()
        stat = DLL.DWOpenDataFile(name.encode(), info)
        if stat:
            raise DWError(stat)
        DWFile.closed = False
        self.name = name
        self.info = info

        # Read file header section
        self.header = dict()
        name_ = ctypes.create_string_buffer(100)
        text_ = ctypes.create_string_buffer(200)
        nHeaders = DLL.DWGetHeaderEntryCount()
        for i in range(nHeaders):
            stat = DLL.DWGetHeaderEntryTextF(i, text_, len(text_))
            if stat:
                raise DWError(stat)
            text = text_.value.decode()
            if len(text) and not(text.startswith('Select...') or 
                    text.startswith('To fill out')):
                stat = DLL.DWGetHeaderEntryNameF(i, name_, len(name_))
                if stat:
                    raise DWError(stat)
                self.header[name_.value.decode()] = text
            
        # Read channel metadata
        nchannels = DLL.DWGetChannelListCount()
        self.channels = (DWChannel * nchannels)()
        stat = DLL.DWGetChannelList(self.channels)
        if stat:
            raise DWError(stat)
            
    def events(self):
        """Load and return timeseries of file events"""
        import pandas
        time_stamp = []
        event_type = []
        event_text = []
        nEvents = DLL.DWGetEventListCount()
        if nEvents:
            events_ = (DWEvent * nEvents)()
            stat = DLL.DWGetEventList(events_)
            if stat:
                raise DWError(stat)
            for e in events_:
                time_stamp.append(e.time_stamp)
                event_type.append(e.event_type)
                event_text.append(e.event_text)
        return pandas.DataFrame(
                data = {'type': event_type, 'text': event_text},
                index = time_stamp)

    def dataframe(self, columns):
        """Return dataframe of selected series"""
        import pandas
        d = {}
        for key in columns:
            d[key] = self[key].series()
        return pandas.DataFrame(d)
        
    @classmethod
    def close(cls):
        DLL.DWCloseDataFile()
        cls.closed = True
        
    def __len__(self):
        return len(self.channels)

    def __getitem__(self, key):
        for ch in self.channels: # brute force lookup
            if ch.index == key or ch.name == key:
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

# Define module methods
def loadDLL(dllName = ''):
    import os
    import platform
    import atexit
    
    global DLL
    if not dllName:
        if platform.architecture()[0] == '32bit':
            dllName = os.path.join(os.path.dirname(__file__),
                "DWDataReaderLib")
        else:
            dllName = os.path.join(os.path.dirname(__file__),
                "DWDataReaderLib64")
    DLL = ctypes.cdll.LoadLibrary(dllName)
    stat = DLL.DWInit()
    if stat:
        raise DWError(stat)
    atexit.register(unloadDLL)
        
def unloadDLL():
    DLL.DWDeInit()

def open(name):
    return DWFile(name)

def close():
    return DWFile.close()

# Load and initialize the DLL
loadDLL()


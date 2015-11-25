"""Python module to interact with Dewesoft DWDataReaderLib.dll

@author: shelmore and costerwisch

Example usage:
import dwdatareader as dw
with dw.open('myfile.d7d') as f:
    print(f.info)
    ch1 = f['chname1'].series()
    ch1.plot()
    for ch in f.values():
        print(ch.name, ch.series().mean())
"""
__all__ = ['DWError', 'DWFile', 'getVersion']
__version__ = '0.9.1'

DLL = None # module variable accessible to other classes 

import os
import collections
import ctypes

class DWError(RuntimeError):
    """Interpret error number returned from dll"""
    errors = ("status OK", "error in DLL", "cannot open d7d file",
            "file already in use", "d7d file corrupt", "memory allocation",
            "creating uncompressed file", "extracting data",
            "opening uncompressed file")

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
        """Readable description of the event"""
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
        """An idenfitying name of the channel"""
        return self._name.decode()

    @property
    def unit(self):
        """The unit of measurement used by the channel"""
        return self._unit.decode()

    @property
    def description(self):
        """A short explanation of what the channel measures"""
        return self._description.decode()

    def __str__(self):
        return "{0.name} ({0.unit}) {0.description}".format(self)

    def scaled(self):
        """Load and return full speed data as [time, data]"""
        import numpy
        count = DLL.DWGetScaledSamplesCount(self.index)
        if count < 0:
            raise IndexError('DWGetScaledSamplesCount({})={} should be non-negative'.format(
                self.index, count))
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
            ctypes.byref(count), ctypes.byref(block_size))
        if stat:
            raise DWError(stat)
        data = (DWReducedValue * count.value)()
        stat = DLL.DWGetReducedValues(self.index, 0, count,
                ctypes.byref(data))
        if stat:
            raise DWError(stat)
        return data

    def series(self):
        """Load and return timeseries of results for channel"""
        import numpy
        import pandas
        time, data = self.scaled()
        if len(data):
            time, ix = numpy.unique(time, return_index=True)
            data = data[ix] # Remove duplicate times
        else:
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

    def __init__(self, source = None):
        self.name = ''      # Name of the open file
        self.closed = True  # bool indicating the current state of the reader
        self.delete = False # Whether to remove file when closed

        num_readers = ctypes.c_int()
        stat = DLL.DWGetNumReaders(ctypes.byref(num_readers))
        if stat:
            raise DWError(stat)
        self.readerID = num_readers.value - 1

        if source:
            self.open(source) # If this fails then the instance is not constructed

        stat = DLL.DWAddReader()  # Add reader to be used by next DWFile instance
        if stat:
            self.close()
            raise DWError(stat)

    def activate(self, verifyOpen=True):
        """Set this DWFile instance as the active reader"""
        if verifyOpen and self.closed:
            raise ValueError('I/O operation on closed file.')
        stat = DLL.DWSetActiveReader(self.readerID)
        if stat:
            raise DWError(stat)

    def open(self, source):
        """Open the specified file and read channel metadata"""

        import tempfile
        self.close() # ensure any previous file has been closed
        self.activate(verifyOpen=False)
        try:
            if hasattr(source, 'read'): # source is a file-like object
                temp_fd, self.name = tempfile.mkstemp(suffix='.d7d') # Create tempfile
                self.delete = True
                with os.fdopen(temp_fd, mode='wb') as ts:
                    ts.write(source.read()) # Make a temporary copy
            else:   # assume source is a str filename
                self.name = source
                self.delete = False

            # Open the d7d file
            self.info = DWInfo()
            stat = DLL.DWOpenDataFile(self.name.encode(), ctypes.byref(self.info))
            if stat:
                raise DWError(stat)
            self.closed = False

            # Read channel metadata
            nchannels = DLL.DWGetChannelListCount()
            self.channels = (DWChannel * nchannels)()
            stat = DLL.DWGetChannelList(self.channels)
            if stat:
                raise DWError(stat)
        except:
            self.close() # if open() fails then the file should be closed
            raise

    @property
    def header(self):
        """Read file header section"""
        self.activate()
        h = dict()
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
                h[name_.value.decode()] = text
        return h

    def events(self):
        """Load and return timeseries of file events"""
        import pandas
        self.activate()
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

    def dataframe(self, channels = None):
        """Return dataframe of selected series"""
        import pandas
        self.activate()
        if not channels:
            # Return dataframe of ALL channels by default
            channels = self.keys()
        return pandas.DataFrame({k: self[k].series() for k in channels})

    def close(self):
        """Close the d7d file and delete it if temporary"""
        if not self.closed:
            # Attempting to close an already closed reader seems to crash the DLL
            self.activate()
            DLL.DWCloseDataFile()
            self.closed = True
            self.channels = (DWChannel * 0)() # Delete channel metadata
        if self.delete:
            os.remove(self.name)
            self.delete = False

    def __len__(self):
        return len(self.channels)

    def __getitem__(self, key):
        for ch in self.channels: # brute force lookup
            if ch.index == key or ch.name == key:
                self.activate()
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

    def __exit__(self, exception_type, exception_value, traceback):
        """Close file when it goes out of context"""
        self.close()


# Define module methods
def loadDLL(dllName = ''):
    import platform
    import atexit

    global DLL
    if not dllName:
        # Determine appropriate library to load
        dllName = os.path.join(os.path.dirname(__file__),
            "DWDataReaderLib")
        if platform.architecture()[0] == '64bit':
            dllName += "64"
        if platform.system() == 'Linux':
            dllName += ".so"
    DLL = ctypes.cdll.LoadLibrary(dllName)
    stat = DLL.DWInit()
    if stat:
        raise DWError(stat)
    atexit.register(unloadDLL)


def getVersion():
    return DLL.DWGetVersion()


def unloadDLL():
    global DLL
    DLL.DWDeInit()
    DLL = None # Release reference to DLL


def open(source):
    return DWFile(source)


# Load and initialize the DLL
loadDLL()


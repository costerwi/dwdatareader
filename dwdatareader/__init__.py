"""Python module that wraps Dewesoft DWDataReaderLib.dll for interactive use with Pyton

@author: shelmore, costerwisch and Bojan023

Example usage:
import dwdatareader as dw
with dw.open('myfile.d7d') as f:
    print(f.info)
    ch1 = f['chname1'].series()
    for ch in f.values():
        print(ch.name, ch.series().mean())
"""

import ctypes
import platform
import atexit
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, List
from enum import IntEnum
from xml.etree import ElementTree

__all__ = ['get_version', 'open_file']
__version__ = '1.0.0'

encoding = 'utf-8'  # default encoding
DLL: getattr(ctypes, "WinDLL", ctypes.CDLL)

class DWArrayInfoStruct(ctypes.Structure):
    """Represents information about an axis on and array channel."""
    _pack_ = 1
    _fields_ = [
        ("index", ctypes.c_int),
        ("_name", ctypes.c_char * 100),
        ("_unit", ctypes.c_char * 20),
        ("size", ctypes.c_int)
    ]

    @property
    def name(self):
        """An idenfitying name of the array"""
        return decode_bytes(self._name)

    @property
    def unit(self):
        """The unit of measurement used by the array"""
        return decode_bytes(self._unit)

class DWArrayInfo(DWArrayInfoStruct):
    def __init__(self, array_struct: DWArrayInfoStruct, channel = None, *args: Any, **kw: Any):
        super().__init__(*args, **kw)
        ctypes.memmove(ctypes.addressof(self), ctypes.addressof(array_struct), ctypes.sizeof(array_struct))
        self.channel = channel

    @property
    def name(self):
        """An identifying name of the array"""
        if self._name == '_':  # this happens on arrays with multiple columns
            return self.colums[self.index]
        else:
            return decode_bytes(self._name)

    @property
    def columns(self):
        """Idenfitying names for columns of a multidimensional array"""
        root = ElementTree.fromstring(self.channel.channel_xml)

        element = root.find('./ArrayInfo/Axis/StringValues').text
        column_names = None

        if element is not None:
            column_names = element.split(';')[1:]

        prefix = self.channel.name
        if column_names is not None:
            return [f'{prefix}_{col_name}' for col_name in column_names]
        else:
            return [self.channel.array_info[self.index].name for _ in range(self.size)]

    def __str__(self):
        return f"DWArrayInfo index={self.index} name='{self.name}' unit='{self.unit}' size={self.size}"

class DWBinarySample(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("position", ctypes.c_longlong),
        ("size", ctypes.c_longlong)
    ]

class DWChannelProps(IntEnum):
    """Specifies the properties that can be retrieved for a channel."""
    DW_DATA_TYPE = 0
    DW_DATA_TYPE_LEN_BYTES = 1
    DW_CH_INDEX = 2
    DW_CH_INDEX_LEN = 3
    DW_CH_TYPE = 4
    DW_CH_SCALE = 5
    DW_CH_OFFSET = 6
    DW_CH_XML = 7
    DW_CH_XML_LEN = 8
    DW_CH_XMLPROPS = 9
    DW_CH_XMLPROPS_LEN = 10
    DW_CH_CUSTOMPROPS = 11
    DW_CH_CUSTOMPROPS_COUNT = 12
    DW_CH_LONGNAME = 13
    DW_CH_LONGNAME_LEN = 14

class DWChannelStruct(ctypes.Structure):
    """Structure represents a Dewesoft channel."""
    _pack_ = 1
    _fields_ = [
        ("index", ctypes.c_int),
        ("_name", ctypes.c_char * 100),
        ("_unit", ctypes.c_char * 20),
        ("_description", ctypes.c_char * 200),
        ("color", ctypes.c_uint),
        ("array_size", ctypes.c_int),
        ("data_type", ctypes.c_int)  # Using DWDataType enum
    ]

    @property
    def name(self):
        """An idenfitying name of the channel"""
        return decode_bytes(self._name)

    @property
    def unit(self):
        """The unit of measurement used by the channel"""
        return decode_bytes(self._unit)

    @property
    def description(self):
        """A short explanation of what the channel measures"""
        return decode_bytes(self._description)

    @property
    def data_type(self):
        """The type of data stored in the channel"""
        return DWDataType(self._data_type)

class DWChannelType(IntEnum):
    """Specifies the type of channel."""
    DW_CH_TYPE_SYNC = 0
    DW_CH_TYPE_ASYNC = 1
    DW_CH_TYPE_SV = 2

class DWChannel(DWChannelStruct):
    def __init__(self, channel_struct: DWChannelStruct, reader_handle, *args: Any, **kw: Any) -> None:
        # super().__init__(*args, **kw)
        # ctypes.memmove(ctypes.addressof(self), ctypes.addressof(channel_struct), ctypes.sizeof(channel_struct))
        # self.reader_handle = reader_handle
        super().__init__(*args, **kw)

        # Create a new instance by copying the buffer memory
        new_struct = self.__class__.from_buffer_copy(bytearray(channel_struct))
        for field_name, _ in self._fields_:
            setattr(self, field_name, getattr(new_struct, field_name))

        self.reader_handle = reader_handle

    @property
    def number_of_samples(self):
        count = ctypes.c_longlong()
        if self.data_type == DWDataType.dtBinary:
            status = DLL.DWIGetBinarySamplesCount(self.reader_handle, self.index, ctypes.byref(count))
        else:
            status = DLL.DWIGetScaledSamplesCount(self.reader_handle, self.index, ctypes.byref(count))
        check_lib_status(status)
        return count.value

    def _chan_prop_int(self, chan_prop):
        count = ctypes.c_longlong(ctypes.sizeof(ctypes.c_int))
        # count = ctypes.c_longlong()
        status = DLL.DWIGetChannelProps(self.reader_handle,
                                      self.index,
                                      ctypes.c_int(chan_prop),
                                      ctypes.byref(count),
                                      ctypes.byref(count))
        check_lib_status(status)
        return count

    def _chan_prop_double(self, chan_prop):
        count = ctypes.c_longlong(ctypes.sizeof(ctypes.c_double))
        # count = ctypes.c_longlong()
        p_buff = ctypes.c_double(0)
        status = DLL.DWIGetChannelProps(self.reader_handle,
                                      self.index, ctypes.c_int(chan_prop), ctypes.byref(p_buff),
                                      ctypes.byref(count))
        check_lib_status(status)
        return p_buff

    def _chan_prop_str(self, chan_prop, chan_prop_len):
        len_str = self._chan_prop_int(chan_prop_len)
        p_buff = ctypes.create_string_buffer(len_str.value)
        status = DLL.DWIGetChannelProps(self.reader_handle,
                                      self.index, ctypes.c_int(chan_prop), p_buff,
                                      ctypes.byref(len_str))
        check_lib_status(status)
        return decode_bytes(p_buff.value)

    @property
    def channel_type(self):
        return DWChannelType(self._chan_prop_int(DWChannelProps.DW_CH_TYPE).value)

    @property
    def channel_index(self):
        return self._chan_prop_str(DWChannelProps.DW_CH_INDEX,
                                   DWChannelProps.DW_CH_INDEX_LEN)

    @property
    def channel_xml(self):
        return self._chan_prop_str(DWChannelProps.DW_CH_XML,
                                   DWChannelProps.DW_CH_XML_LEN)

    @property
    def long_name(self):
        return self._chan_prop_str(DWChannelProps.DW_CH_LONGNAME,
                                   DWChannelProps.DW_CH_LONGNAME_LEN)

    @property
    def scale(self):
        return self._chan_prop_double(DWChannelProps.DW_CH_SCALE).value

    @property
    def offset(self):
        return self._chan_prop_double(DWChannelProps.DW_CH_OFFSET).value

    @property
    def array_info(self):
        """Return list of DWArrayInfo axes for this channel"""
        if self.array_size < 2:
            return []
        narray_infos = ctypes.c_longlong()
        status = DLL.DWIGetArrayInfoCount(self.reader_handle, self.index, ctypes.byref(narray_infos)) # available array axes for this channel
        check_lib_status(status)
        if narray_infos.value < 1:
            raise IndexError(f'DWIGetArrayInfoCount({self.index})={narray_infos} should be >0')
        axes = (DWArrayInfoStruct * narray_infos.value)()
        status = DLL.DWIGetArrayInfoList(self.reader_handle, self.index, axes)
        check_lib_status(status)

        axes = [DWArrayInfo(ax, self) for ax in axes]

        return axes

    def __str__(self):
        return f"{self.name} ({self.unit}) {self.description}"

    def scaled(self, array_index=0):
        """Load and return full speed data"""
        if not 0 <= array_index < self.array_size:
            raise IndexError('array index is out of range')
        count = self.number_of_samples
        data = np.zeros(count*self.array_size, dtype=np.double)
        time = np.zeros(count, dtype=np.double)
        status = DLL.DWIGetScaledSamples(self.reader_handle, self.index,
                                       ctypes.c_longlong(0), ctypes.c_longlong(self.number_of_samples),
                                       data.ctypes, time.ctypes)
        check_lib_status(status)

        return time, data

    def dataframe(self):
        """Load and return full speed channel data as Pandas Dataframe"""
        if self.data_type == DWDataType.dtBinary:
            sample_cnt = self.number_of_samples

            assert self.channel_type == DWChannelType.DW_CH_TYPE_ASYNC
            assert self.array_size == 1

            timestamps = (ctypes.c_double * sample_cnt)()
            data = (DWBinarySample * sample_cnt)()
            status = DLL.DWIGetBinRecSamples(self.reader_handle, self.index, ctypes.c_longlong(0), sample_cnt, data,
                                             timestamps)
            check_lib_status(status)

            bin_buf_size = 1024
            parsed_data = []
            for i in range(sample_cnt):
                bin_rec = data[i]
                bin_buf = create_string_buffer(bin_buf_size)
                bin_buf_pos = ctypes.c_longlong(0)
                status = DLL.DWIGetBinData(
                    self.reader_handle, self.index,
                    bin_rec, bin_buf,
                    ctypes.byref(bin_buf_pos), bin_buf_size
                )
                check_lib_status(status)
                # Append timestamp and decoded binary data
                parsed_data.append({
                    "Timestamp": timestamps[i],
                    "Value": decode_bytes(bin_buf.value)
                })

            # Return as a Pandas DataFrame
            return pd.DataFrame(parsed_data)
        else:
            time, data = self.scaled()

            columns = []
            if self.array_size == 1:
                columns.append(self.name)
            else:  # Channel has multiple axes
                for array_info in self.array_info:
                    columns.extend(array_info.columns)

            time, ix = np.unique(time, return_index=True)  # unique times required for reindexing
            df = pd.DataFrame(
                data=data.reshape(self.number_of_samples, self.array_size)[ix,:],
                index=time,
                columns=columns)

            return df

    def series(self):
        """Load and return timeseries for a channel"""
        time, data = self.scaled()
        return pd.Series(data, index=time)

    def series_generator(self, chunk_size, array_index:int = 0):
        """Generator yielding channel data as chunks of a pandas series

        :param chunk_size: length of chunked series
        :type array_index: int
        :returns: pandas.Series
        """
        count = self.number_of_samples
        chunk_size = min(chunk_size, count)
        data = np.zeros(chunk_size*self.array_size, dtype=np.double)
        time = np.zeros(chunk_size)
        for chunk in range(0, count, chunk_size):
            chunk_size = min(chunk_size, count - chunk)
            status = DLL.DWIGetScaledSamples(
                self.reader_handle,
                self.index,
                ctypes.c_longlong(chunk), ctypes.c_longlong(chunk_size),
                data.ctypes, time.ctypes)
            check_lib_status(status)

            time, ix = np.unique(time[:chunk_size], return_index=True)
            yield pd.Series(
                    data = data.reshape(-1, self.array_size)[ix, array_index],
                    index = time,
                    name = self.name)

    def reduced(self):
        """Load reduced (averaged) data as Pandas DataFrame"""
        count = ctypes.c_longlong()
        block_size = ctypes.c_double()
        status = DLL.DWIGetReducedValuesCount(self.reader_handle, self.index,
            ctypes.byref(count), ctypes.byref(block_size))
        check_lib_status(status)

        # Define a numpy structured data type to hold DWReducedValue
        reduced_dtype = np.dtype([
                ('time_stamp', np.double),
                ('ave', np.double),
                ('min', np.double),
                ('max', np.double),
                ('rms', np.double)
            ])

        # Allocate memory and retrieve data
        data = np.empty(count.value, dtype=reduced_dtype)
        status = DLL.DWIGetReducedValues(self.reader_handle, self.index, 0, count, data.ctypes)
        check_lib_status(status)

        return pd.DataFrame(data, index=data['time_stamp'],
                columns=['ave', 'min', 'max', 'rms'])

class DWComplex(ctypes.Structure):
    """Represents a complex number with real and imaginary components."""
    _pack_ = 1
    _fields_ = [
        ("re", ctypes.c_double),
        ("im", ctypes.c_double)
    ]

class DWDataType(IntEnum):
    """Specifies the channel data type."""
    dtByte = 0
    dtShortInt = 1
    dtSmallInt = 2
    dtWord = 3
    dtInteger = 4
    dtSingle = 5
    dtInt64 = 6
    dtDouble = 7
    dtLongword = 8
    dtComplexSingle = 9
    dtComplexDouble = 10
    dtText = 11
    dtBinary = 12
    dtCANPortData = 13
    dtCANFDPortData = 14
    dtBytes8 = 15
    dtBytes16 = 16
    dtBytes32 = 17
    dtBytes64 = 18

class DWEvent(ctypes.Structure):
    """Represents an event in a datafile."""
    _pack_ = 1
    _fields_ = [
        ("_event_type", ctypes.c_int),
        ("time_stamp", ctypes.c_double), # timestamp in seconds relative to start_measure_time
        ("_event_text", ctypes.c_char * 200)
    ]

    @property
    def event_type(self):
        return DWEvent(self._event_type)

    @property
    def event_text(self):
        """Readable description of the event"""
        return decode_bytes(self._event_text)


    def __str__(self):
        return f"{self.event_type} {self.time_stamp} {self.event_text}"

class DWFile(dict):
    """Data file type mapping channel names their metadata"""
    def __init__(self, source = None):
        self.name = ''      # Name of the open file
        self.closed = True  # bool indicating the current state of the reader
        self.info = None

        global DLL
        DLL = load_library()

        reader_handle = ctypes.c_void_p()
        status = DLL.DWICreateReader(ctypes.byref(reader_handle))
        check_lib_status(status)
        self.reader_handle = reader_handle
        atexit.register(self.close)  # for interpreter shutdown

        if source:
            self.open(source)

    def open(self, source):
        """Open the specified file and read channel metadata"""
        try:
            # Open the d7d file
            self.info = DWMeasurementInfo()
            c_source = ctypes.c_char_p(source.encode(encoding=encoding))
            # DWIOpenDataFile outputs DWFileInfo struct, however DWFile is marked as deprecated
            status = DLL.DWIOpenDataFile(self.reader_handle, c_source, ctypes.byref(self.info))
            check_lib_status(status)

            # fill all DWMeasurementInfo fields not filled by DWIOpenDataFile
            status = DLL.DWIGetMeasurementInfo(self.reader_handle, ctypes.byref(self.info))
            check_lib_status(status)
            self.closed = False

            # Read channel metadata
            ch_count = ctypes.c_longlong()
            status = DLL.DWIGetChannelListCount(self.reader_handle, ctypes.byref(ch_count))
            check_lib_status(status)
            channel_structs = (DWChannelStruct * ch_count.value)()

            status = DLL.DWIGetChannelList(self.reader_handle, channel_structs)
            check_lib_status(status)

            self.update([ch.name, DWChannel(ch, self.reader_handle)] for ch in channel_structs)

            # read binary channel metadata
            bin_ch_count = ctypes.c_longlong()
            status = DLL.DWIGetBinChannelListCount(self.reader_handle, ctypes.byref(bin_ch_count))
            check_lib_status(status)
            bin_channel_structs = (DWChannelStruct * bin_ch_count.value)()

            status = DLL.DWIGetBinChannelList(self.reader_handle, bin_channel_structs)
            check_lib_status(status)

            self.update([ch.name, DWChannel(ch, self.reader_handle)] for ch in bin_channel_structs)

        except RuntimeError as e:
            print(e)
            self.close()
            raise

    @property
    def header(self):
        """Read file header section"""
        header = dict()
        name_ = ctypes.create_string_buffer(100)
        text_ = ctypes.create_string_buffer(200)
        count = ctypes.c_longlong()
        status = DLL.DWIGetHeaderEntryCount(self.reader_handle, ctypes.byref(count))
        check_lib_status(status)
        for i in range(count.value):
            status = DLL.DWIGetHeaderEntryTextF(self.reader_handle, i, text_, len(text_))
            check_lib_status(status)
            text = decode_bytes(text_.value)
            if len(text) and not(text.startswith('Select...') or
                    text.startswith('To fill out')):
                status = DLL.DWIGetHeaderEntryNameF(self.reader_handle, i, name_, len(name_))
                check_lib_status(status)
                header[decode_bytes(name_.value)] = text
        return header

    @property
    def storing_type(self):
        storing_type = ctypes.c_int()
        status = DLL.DWIGetStoringType(self.reader_handle, ctypes.byref(storing_type))
        check_lib_status(status)
        return DWStoringType(storing_type.value)

    def export_header(self, file_name):
        """Export header as .xml file"""
        c_file_name = ctypes.c_char_p(file_name.encode(encoding=encoding))
        status = DLL.DWIExportHeader(self.reader_handle, c_file_name)
        check_lib_status(status)
        return 0

    def events(self):
        """Load and return timeseries of file events"""
        time_stamp = []
        event_type = []
        event_text = []
        count = ctypes.c_longlong()
        status = DLL.DWIGetEventListCount(self.reader_handle, ctypes.byref(count))
        check_lib_status(status)
        if count.value:
            events_ = (DWEvent * count.value)()
            status = DLL.DWIGetEventList(self.reader_handle, events_)
            check_lib_status(status)
            for e in events_:
                time_stamp.append(e.time_stamp)
                event_type.append(e.event_type)
                event_text.append(e.event_text)
        return pd.DataFrame(
                data = {'type': event_type, 'text': event_text},
                index = time_stamp)

    def dataframe(self, channels: List = None) -> pd.DataFrame:
        """Return dataframe of selected channels"""
        if channels is None:
            # Return dataframe of all channels by default
            channels = [ch.name for ch in self.values() if ch.channel_type == DWChannelType.DW_CH_TYPE_SYNC]

        channel_dfs = [self[ch_name].dataframe() for ch_name in channels]
        df = pd.concat(channel_dfs, axis=1, sort=True, copy=False)
        return df

    def close(self):
        """Close the d7d file and delete it if temporary"""
        if not self.closed:
            DLL.DWICloseDataFile(self.reader_handle)
            if self.reader_handle is not None:
                DLL.DWIDestroyReader(self.reader_handle)
                self.reader_handle = None
            self.closed = True
            self.clear()  # Delete channel metadata

    def __str__(self):
        return self.name

    def __enter__(self):
        """Used to maintain file in context"""
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Close a file when it goes out of context"""
        self.close()

    def __del__(self):  # object destruction
        self.close()

class DWMeasurementInfo(ctypes.Structure):
    """Structure with information about the current measurement."""
    _pack_ = 1
    _fields_ = [
        ("sample_rate", ctypes.c_double),
        ("_start_measure_time", ctypes.c_double),
        ("_start_store_time", ctypes.c_double),
        ("duration", ctypes.c_double)
    ]

    def __str__(self):
        return f"{self.sample_rate} Hz | {self.start_measure_time} | {self.start_store_time} | {self.duration} s"

    @property
    def start_store_time(self):
        """Return start_store_time in Python datetime format"""
        epoch = datetime(1899, 12, 30, tzinfo=timezone.utc)
        return epoch + timedelta(self._start_store_time)

    @property
    def start_measure_time(self):
        """Return start_store_time in Python datetime format"""
        epoch = datetime(1899, 12, 30, tzinfo=timezone.utc)
        return epoch + timedelta(self._start_store_time)

class DWStatus(IntEnum):
    """Status codes returned from library function calls"""
    DWSTAT_OK = 0
    DWSTAT_ERROR = 1
    DWSTAT_ERROR_FILE_CANNOT_OPEN = 2
    DWSTAT_ERROR_FILE_ALREADY_IN_USE = 3
    DWSTAT_ERROR_FILE_CORRUPT = 4
    DWSTAT_ERROR_NO_MEMORY_ALLOC = 5
    DWSTAT_ERROR_CREATE_DEST_FILE = 6
    DWSTAT_ERROR_EXTRACTING_FILE = 7
    DWSTAT_ERROR_CANNOT_OPEN_EXTRACTED_FILE = 8
    DWSTAT_ERROR_INVALID_IB_LEVEL = 9
    DWSTAT_ERROR_CAN_NOT_SUPPORTED = 10
    DWSTAT_ERROR_INVALID_READER = 11
    DWSTAT_ERROR_INVALID_INDEX = 12
    DWSTAT_ERROR_INSUFFICENT_BUFFER = 13

class DWStoringType(IntEnum):
    """Specifies the type data storing mode."""
    ST_ALWAYS_FAST = 0
    ST_ALWAYS_SLOW = 1
    ST_FAST_ON_TRIGGER = 2
    ST_FAST_ON_TRIGGER_SLOW_OTH = 3

def get_version():
    ver_major = ctypes.c_int()
    ver_minor = ctypes.c_int()
    ver_patch = ctypes.c_int()
    DLL = load_library()
    DLL.DWGetVersionEx(ctypes.byref(ver_major), ctypes.byref(ver_minor), ctypes.byref(ver_patch))

    return f"{ver_major.value}.{ver_minor.value}.{ver_patch.value}"

def open_file(source):
    return DWFile(source)

def load_library(custom_path=None):
    """
    Load a shared or dynamic library based on platform and architecture.
    :param custom_path: Optional custom path to load the library from.
    :return: Loaded library object.
    """
    library_extensions = {
        "Linux": ".so",
        "Darwin": ".dylib",
        "Windows": ".dll",
    }

    system = platform.system()
    is_64bit = platform.architecture()[0] == '64bit'

    loader = ctypes.CDLL if system != "Windows" else ctypes.WinDLL
    if custom_path is not None:
        return loader(custom_path)

    base_name = "DWDataReaderLib"
    arch_suffix = "64" if is_64bit else ""
    extension = library_extensions.get(system, "")

    library_name = f"{base_name}{arch_suffix}{extension}"
    library_path = Path(__file__).parent / library_name

    try:
        # Load the library
        return loader(str(library_path))
    except OSError:
        # Fallback to direct library name loading
        return loader(library_name)

def create_string_buffer(string_value, buffer_size=None):
    """Create a string buffer with proper encoding."""
    if isinstance(string_value, str):
        return ctypes.create_string_buffer(string_value.encode(encoding=encoding), buffer_size)
    return ctypes.create_string_buffer(string_value, buffer_size)

def decode_bytes(byte_string):
    """Convert bytes to string with proper decoding."""
    if isinstance(byte_string, bytes):
        return byte_string.decode(encoding=encoding, errors='replace').rstrip('\x00')
    return byte_string

def check_lib_status(status: DWStatus):
    """Check the status returned by the library functions."""
    if status == DWStatus.DWSTAT_OK:
        return

    err_msg_len = ctypes.c_int(1024)
    err_msg = create_string_buffer(err_msg_len.value)
    err_status = ctypes.c_int(DWStatus.DWSTAT_OK)

    while DLL.DWGetLastStatus(ctypes.byref(err_status), err_msg,
                              ctypes.byref(err_msg_len)) == DWStatus.DWSTAT_ERROR_NO_MEMORY_ALLOC:
        err_msg = create_string_buffer(err_msg_len.value)

    raise RuntimeError(f"Error {status}: {decode_bytes(err_msg.value)}")

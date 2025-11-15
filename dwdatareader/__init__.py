"""Python module that wraps Dewesoft DWDataReaderLib.dll for interactive use with Pyton

Homepage: https://github.com/costerwi/dwdatareader/

Example usage:
import dwdatareader as dw
with dw.DWFile('myfile.d7d') as f:
    print(f.info)
    ch1 = f['chname1'].series()
    for ch in f.values():
        print(ch.name, ch.series().mean())
"""

__all__ = ['get_version', 'open_file', 'DWError', 'DWFile']
__version__ = '1.1.0'

import ctypes
import platform
import atexit
from datetime import datetime, timezone, timedelta
from pathlib import Path
from enum import IntEnum
from xml.etree import ElementTree
from typing import Any, Callable, List, Tuple, Optional

import numpy as np
import pandas as pd

encoding = 'utf-8'  # default encoding

class DWArrayInfoStruct(ctypes.Structure):
    """
    A structure for information of an array.

    This class is a ctypes Structure to define metadata for an array. The
    structure is compliant with relevant DLL function calls.

    Properties:
        index (int): index
        name (char): name
        unit (char: measurement unit
        size (int): array size (number of dimensions)
    """
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
    """
    Represents information about an array.

    This class inherits from DWArrayInfoStruct and encapsulates information of
    a data array. Instances of this class provide mechanisms to retrieve relevant
    array metadata.

    Attributes:
        channel: Reference to the channel associated with this array. This is
                 used to access additional data or metadata for the array.
    """
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
        """Extracting idenfitying names from array XML for columns of a multidimensional array"""
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
    """
    Represents a binary sample.

    This class is a ctypes Structure to define binary samples. The
    structure is compliant with relevant DLL function calls.

    Properties:
        position (int): position of 4.
        name (char): name of the array.
        unit (char: The unit of measurement used by the array.
        size (int): The size of the array.
    """
    _pack_ = 1
    _fields_ = [
        ("position", ctypes.c_longlong),
        ("size", ctypes.c_longlong)
    ]

class DWChannelProps(IntEnum):
    """
    Defines an enumeration for channel properties.

    This class is an enumeration that specifies various properties
    pertaining to channels. The keys represent the property names
    and values are associated integers.

    Properties:
        DW_DATA_TYPE (int): data type of the channel
        DW_DATA_TYPE_LEN_BYTES (int): length of the data type in bytes
        DW_CH_INDEX (int): channel index
        DW_CH_INDEX_LEN (int): length of the channel index
        DW_CH_TYPE (int): channel type
        DW_CH_SCALE (int): scale factor of the channel values
        DW_CH_OFFSET (int): offset of the channel values
        DW_CH_XML (int): XML structure of channel
        DW_CH_XML_LEN (int): length of the channel XML data
        DW_CH_XMLPROPS (int): XML structure properties
        DW_CH_XMLPROPS_LEN (int): Length of XML structure properties
        DW_CH_CUSTOMPROPS (int): XML structure custom properties
        DW_CH_CUSTOMPROPS_COUNT (int): length of XML structure custom properties
        DW_CH_LONGNAME (int): long name of the channel
        DW_CH_LONGNAME_LEN (int): length of the long name
    """
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
    """
    Represents the structure for a channel.

    This class is a ctypes Structure to store a channel's metadata. The
    structure is compliant with relevant DLL function calls.

    Properties:
        index: unique index of the channel
        name: channel name
        unit: measurement unit
        description: channel description
        color: display color
        array_size: array size (dimension)
        data_type: type of data stored in the channel, refers to DWDataType
    """
    _pack_ = 1
    _fields_ = [
        ("index", ctypes.c_int),
        ("_name", ctypes.c_char * 100),
        ("_unit", ctypes.c_char * 20),
        ("_description", ctypes.c_char * 200),
        ("color", ctypes.c_uint),
        ("array_size", ctypes.c_int),
        ("_data_type", ctypes.c_int)
    ]

    @property
    def name(self):
        """An idenfitying name of the channel"""
        return decode_bytes(self._name)

    @property
    def unit(self):
        """The unit of measurement used by the channel"""
        return decode_bytes(self._unit)

    def get_scaled_samples(self):
        if self.data_type == DWDataType.dtBinary:
            return DLL.DWIGetBinarySamples
        if self._is_complex():
            return DLL.DWIGetComplexScaledSamples
        else:
            return DLL.DWIGetScaledSamples

    @property
    def description(self):
        """A short explanation of what the channel measures"""
        return decode_bytes(self._description)
    @property
    def data_type(self):
        """The type of data stored in the channel"""
        return DWDataType(self._data_type)

class DWChannelType(IntEnum):
    """
    Represents a channel type.

    The DWChannelType enumeration translates integers to different
    types of channels, primarily used to categorize behavior in terms
    of synchronization, value handling or other attributes.

    Properties:
        DW_CH_TYPE_SYNC: synchronous channel
        DW_CH_TYPE_ASYNC: asynchronous channel
        DW_CH_TYPE_SV: single value channel
    """
    DW_CH_TYPE_SYNC = 0
    DW_CH_TYPE_ASYNC = 1
    DW_CH_TYPE_SV = 2

class DWChannel(DWChannelStruct):
    """
    Represents a data channel, providing methods to access its data.

    This class inherits from DWChannelStruct to provide access to its properties
    and data. It interacts with the DLL and provides utility methods to handle
    channel data of various data types.

    Attributes:
        dwFile: reference to the DWFile containing this channel

    Methods:
        number_of_samples: number of samples in the channel
        channel_type: channel type, refering to DWChannelType
        channel_index: channel index
        channel_xml: XML of the channel configuration
        long_name: long name of the channel
        scale: scale factor applied to the channel data
        offset: offset applied to the channel data
        array_info: DWArrayInfo axes associated with the channel
        scaled: full speed data and timestamps as numpy arrays
        dataframe: full speed data as a Pandas DataFrame
        series: channel data as a Pandas Series
        series_generator: channel data in chunked Pandas Series format
        reduced: reduced (averaged) data as a Pandas DataFrame
    """

    def __init__(self):
        super().__init__()
        self.dwFile = None

    @property
    def dtype(self):
        if self._is_complex():
            return np.complex128
        return np.double

    def _chan_prop_int(self, chan_prop):
        prop_int = ctypes.c_longlong(ctypes.sizeof(ctypes.c_int))
        status = DLL.DWIGetChannelProps(self.dwFile.reader_handle,
                                      self.index,
                                      ctypes.c_int(chan_prop),
                                      ctypes.byref(prop_int),
                                      ctypes.byref(prop_int))
        if status: raise DWError(status)
        return prop_int

    def _chan_prop_double(self, chan_prop: int) -> ctypes.c_double:
        """
        Retrieves the floating point value for a specific channel property.
        This method interacts with the DLL to fetch the floating point value
        associated with a given channel property for the associated reader.

        Parameters:
        chan_prop (int): identifier for the channel property

        Returns:
        count (ctypes.c_double): floating point value for the channel property
        """
        count = ctypes.c_longlong(ctypes.sizeof(ctypes.c_double))
        prop_double = ctypes.c_double(0)
        status = DLL.DWIGetChannelProps(self.dwFile.reader_handle,
                                      self.index, ctypes.c_int(chan_prop), ctypes.byref(prop_double),
                                      ctypes.byref(count))
        if status: raise DWError(status)
        return prop_double

    def _chan_prop_str(self, chan_prop, chan_prop_len):
        """
        Retrieves the string for a specific channel property.

        This method interacts with the DLL to fetch the string associated
        with a given channel property for the associated reader and decodes it.

        Parameters:
        chan_prop (int): identifier for the channel property

        Returns:
        count (str): string for the channel property
        """
        len_str = self._chan_prop_int(chan_prop_len)
        str_buff = ctypes.create_string_buffer(len_str.value)
        status = DLL.DWIGetChannelProps(self.dwFile.reader_handle,
                                      self.index, ctypes.c_int(chan_prop), str_buff,
                                      ctypes.byref(len_str))
        if status: raise DWError(status)
        return decode_bytes(str_buff.value)

    @property
    def number_of_samples(self):
        count = ctypes.c_longlong()
        if self.data_type == DWDataType.dtBinary:
            status = DLL.DWIGetBinarySamplesCount(self.dwFile.reader_handle, self.index, ctypes.byref(count))
        if self._is_complex():
            status = DLL.DWIGetComplexScaledSamplesCount(self.dwFile.reader_handle, self.index, ctypes.byref(count))
        else:
            status = DLL.DWIGetScaledSamplesCount(self.dwFile.reader_handle, self.index, ctypes.byref(count))
        if status: raise DWError(status)
        return count.value

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
        status = DLL.DWIGetArrayInfoCount(self.dwFile.reader_handle, self.index, ctypes.byref(narray_infos)) # available array axes for this channel
        if status: raise DWError(status)
        if narray_infos.value < 1:
            raise IndexError(f'DWIGetArrayInfoCount({self.index})={narray_infos} should be >0')
        axes = (DWArrayInfoStruct * narray_infos.value)()
        status = DLL.DWIGetArrayInfoList(self.dwFile.reader_handle, self.index, axes)
        if status: raise DWError(status)

        axes = [DWArrayInfo(ax, self) for ax in axes]

        return axes

    def __str__(self):
        return f"{self.name} ({self.unit}) {self.description}"

    def __repr__(self):
        return self.__str__()

    def scaled(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves scaled channel values with their corresponding timestamps.

        Returns:
            time, data (Tuple[np.ndarray, np.ndarray]): A tuple containing the timestamps as the
                first element and the scaled data array as the second element.
        """
        count = self.number_of_samples
        data = np.zeros(count*self.array_size, dtype=self.dtype())
        time = np.zeros(count, dtype=np.double)
        status = self.get_scaled_samples()(
            self.dwFile.reader_handle,
            self.index,
            ctypes.c_longlong(0),
            ctypes.c_longlong(count),
            data.ctypes,
            time.ctypes,
        )
        if status: raise DWError(status)

        return time, data

    def _is_complex(self) -> pd.DataFrame:
        return self.data_type == DWDataType.dtComplexSingle or self.data_type == DWDataType.dtComplexDouble

    def dataframe(self) -> pd.DataFrame:
        """
        Retrieves scaled channel values with their corresponding timestamps in a Pandas DataFrame.

        Returns:
            df (pandas.DataFrame): A Pandas DataFrame containing scaled data with
                                    the timestamps as index
        """
        if self.data_type == DWDataType.dtBinary:
            sample_cnt = self.number_of_samples

            assert self.channel_type == DWChannelType.DW_CH_TYPE_ASYNC
            assert self.array_size == 1

            timestamps = (ctypes.c_double * sample_cnt)()
            bin_samples = (DWBinarySample * sample_cnt)()
            status = DLL.DWIGetBinRecSamples(self.dwFile.reader_handle, self.index,
                    0, sample_cnt, bin_samples, timestamps)
            if status: raise DWError(status)

            bin_data = []
            bin_buf = ctypes.create_string_buffer(1024)
            bin_buf_pos = ctypes.c_longlong(0)
            for bin_rec in bin_samples:
                status = DLL.DWIGetBinData(
                    self.dwFile.reader_handle, self.index,
                    ctypes.byref(bin_rec), ctypes.byref(bin_buf),
                    ctypes.byref(bin_buf_pos), len(bin_buf)
                )
                if status: raise DWError(status)
                bin_data.append(decode_bytes(bin_buf.value))

            # Return as a Pandas DataFrame
            return pd.DataFrame({self.unique_key: bin_data}, index=np.array(timestamps))
        else:
            time, data = self.scaled()

            columns = []
            if self.array_size == 1:
                columns.append(self.unique_key)
            else:  # Channel has multiple axes
                for array_info in self.array_info:
                    columns.extend(array_info.columns)
                data = data.reshape(self.number_of_samples, self.array_size)

            df = pd.DataFrame(
                data=data,
                index=time,
                columns=columns)

            return df

    def series(self):
        """
        Retrieves scaled channel values with their corresponding timestamps in a Pandas Series.

        Returns:
            df (pandas.Series): A Pandas Series containing scaled data with
                                    the timestamps as index
        """
        time, data = self.scaled()
        return pd.Series(data, index=time, name=self.unique_key)

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
                self.dwFile.reader_handle,
                self.index,
                ctypes.c_longlong(chunk), ctypes.c_longlong(chunk_size),
                data.ctypes, time.ctypes)
            if status: raise DWError(status)

            time, ix = np.unique(time[:chunk_size], return_index=True)
            yield pd.Series(
                    data = data.reshape(-1, self.array_size)[ix, array_index],
                    index = time,
                    name = self.unique_key)

    def reduced(self):
        """Load reduced (averaged) data as Pandas DataFrame"""
        count = ctypes.c_longlong()
        block_size = ctypes.c_double()
        status = DLL.DWIGetReducedValuesCount(self.dwFile.reader_handle, self.index,
            ctypes.byref(count), ctypes.byref(block_size))
        if status: raise DWError(status)

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
        status = DLL.DWIGetReducedValues(self.dwFile.reader_handle, self.index, 0, count, data.ctypes)
        if status: raise DWError(status)

        return pd.DataFrame(data, index=data['time_stamp'],
                columns=['ave', 'min', 'max', 'rms'])

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
        ("time_stamp", ctypes.c_double),  # timestamp in seconds relative to start_measure_time
        ("_event_text", ctypes.c_char * 200)
    ]

    @property
    def event_type(self):
        return DWEventType(self._event_type)

    @property
    def event_text(self):
        """Readable description of the event"""
        return decode_bytes(self._event_text)

    def __str__(self):
        return f"{self.event_type} {self.time_stamp} {self.event_text}"

class DWEventType(IntEnum):
    """Specifies the type of event."""
    etStart = 1
    etStop = 2
    etTrigger = 3
    etVStart = 11
    etVStop = 12
    etKeyboard = 20
    etNotice = 21
    etVoice = 22
    etPicture = 23
    etModule = 24
    etAlarm = 25
    etCursorInfo = 26
    etAlarmLevel = 27

class DWError(RuntimeError):
    """Interpret error number returned from DLL"""

    def __init__(self, status: int):
        self.status = DWStatus(status)
        super(DWError, self).__init__(self.status)
        # Request error message string from DLL
        err_msg_len = ctypes.c_int(1024)
        err_msg = ctypes.create_string_buffer(err_msg_len.value)
        err_status = ctypes.c_int(self.status.value)

        DLL.DWGetLastStatus(ctypes.byref(err_status), err_msg,
                                  ctypes.byref(err_msg_len))
        self.message = decode_bytes(err_msg.value[:err_msg_len.value])
        if hasattr(self, "add_note"):  # added in python 3.11
            self.add_note(self.message)

class DWFile(dict):
    """Data file type mapping channel names their metadata"""
    def __init__(self, source: Optional[str]=None, key: Optional[Callable]=None):
        """
        Parameters:
        source (str): optional file name to open
        key (callable): optional function which takes a DWChannel parameter and returns its key
               default: lambda channel: channel.long_name

        Members:
        name (str): Name of the open file
        closed (bool): Whether the file is closed
        """

        self.name = ''      # Name of the open file
        self.closed = True  # bool indicating the current state of the reader

        self.reader_handle = ctypes.c_void_p(None)
        atexit.register(self.close)  # for interpreter shutdown

        if key is None:
            # Default is to use channel.long_name
            self.key = lambda channel: channel.long_name
        else:
            assert callable(key), "The key parameter should take a DWChannel parameter and return its key"
            self.key = key

        if source:
            self.open(source)

    def open(self, source: str):
        """Open the specified file and read channel metadata"""

        if not self.closed:
            self.close()

        # Create a reader_handle for this file
        status = DLL.DWICreateReader(ctypes.byref(self.reader_handle))
        if status: raise DWError(status)

        try:
            # Open the d7d file
            self.info = DWMeasurementInfo()
            c_source = ctypes.c_char_p(source.encode(encoding=encoding))
            # DWIOpenDataFile outputs DWFileInfo struct, however DWFile is marked as deprecated
            status = DLL.DWIOpenDataFile(self.reader_handle, c_source, ctypes.byref(self.info))
            if status: raise DWError(status)
            self.name = source

            # fill all DWMeasurementInfo fields not filled by DWIOpenDataFile
            status = DLL.DWIGetMeasurementInfo(self.reader_handle, ctypes.byref(self.info))
            if status: raise DWError(status)
            self.closed = False

            def add(channel: DWChannel):
                "Add the given channel to this DWFile dict using a unique key"
                channel.dwFile = self
                key = self.key(channel)
                unique_key = key  # start with the key itself
                suffix = 0
                while unique_key in self:
                    suffix -= 1
                    unique_key = f'{key}{suffix}'
                channel.unique_key = unique_key
                self[unique_key] = channel

            # Read channel metadata
            ch_count = ctypes.c_longlong()
            status = DLL.DWIGetChannelListCount(self.reader_handle, ctypes.byref(ch_count))
            if status: raise DWError(status)
            channel_array = (DWChannel * ch_count.value)()

            status = DLL.DWIGetChannelList(self.reader_handle, channel_array)
            if status: raise DWError(status)

            for channel in channel_array:
                add(channel)

            # read complex channel metadata
            ch_count_complex = ctypes.c_longlong()
            status = DLL.DWIGetComplexChannelListCount(self.reader_handle, ctypes.byref(ch_count_complex))
            if status: raise DWError(status)
            channel_structs_complex = (DWChannel * ch_count_complex.value)()

            status = DLL.DWIGetComplexChannelList(self.reader_handle, channel_structs_complex)
            if status: raise DWError(status)

            for channel in channel_structs_complex:
                add(channel)

            # read binary channel metadata
            bin_ch_count = ctypes.c_longlong()
            status = DLL.DWIGetBinChannelListCount(self.reader_handle, ctypes.byref(bin_ch_count))
            if status: raise DWError(status)
            channel_array = (DWChannel * bin_ch_count.value)()

            status = DLL.DWIGetBinChannelList(self.reader_handle, channel_array)
            if status: raise DWError(status)

            for channel in channel_array:
                add(channel)

        except RuntimeError:
            self.close()
            raise

    @property
    def sync_channels(self):
        return [key for key, ch in self.items() if ch.channel_type == DWChannelType.DW_CH_TYPE_SYNC]

    @property
    def async_channels(self):
        return [key for key, ch in self.items() if ch.channel_type == DWChannelType.DW_CH_TYPE_ASYNC]

    @property
    def header(self):
        """Read file header section"""
        header = dict()
        name_ = ctypes.create_string_buffer(100)
        text_ = ctypes.create_string_buffer(200)
        count = ctypes.c_longlong()
        status = DLL.DWIGetHeaderEntryCount(self.reader_handle, ctypes.byref(count))
        if status: raise DWError(status)
        for i in range(count.value):
            status = DLL.DWIGetHeaderEntryTextF(self.reader_handle, i, text_, len(text_))
            if status: raise DWError(status)
            text = decode_bytes(text_.value)
            if len(text) and not(text.startswith('Select...') or
                    text.startswith('To fill out')):
                status = DLL.DWIGetHeaderEntryNameF(self.reader_handle, i, name_, len(name_))
                if status: raise DWError(status)
                header[decode_bytes(name_.value)] = text
        return header

    @property
    def storing_type(self):
        storing_type = ctypes.c_int()
        status = DLL.DWIGetStoringType(self.reader_handle, ctypes.byref(storing_type))
        if status: raise DWError(status)
        return DWStoringType(storing_type.value)

    def export_header(self, file_name):
        """Export header as .xml file"""
        c_file_name = ctypes.c_char_p(file_name.encode(encoding=encoding))
        status = DLL.DWIExportHeader(self.reader_handle, c_file_name)
        if status: raise DWError(status)
        return 0

    def events(self):
        """Load and return timeseries of file events"""
        time_stamp = []
        event_type = []
        event_text = []
        nEvents = ctypes.c_longlong()
        status = DLL.DWIGetEventListCount(self.reader_handle, ctypes.byref(nEvents))
        if status: raise DWError(status)
        if nEvents.value:
            events_ = (DWEvent * nEvents.value)()
            status = DLL.DWIGetEventList(self.reader_handle, events_)
            if status: raise DWError(status)
            for e in events_:
                time_stamp.append(e.time_stamp)
                event_type.append(e.event_type)
                event_text.append(e.event_text)
        return pd.DataFrame(
                data = {'type': event_type, 'text': event_text},
                index = time_stamp)

    def _build_dataframe(self, channels: List = []) -> pd.DataFrame:
        if not channels:
            return pd.DataFrame()

        channel_dfs = [self[ch_name].dataframe() for ch_name in channels]
        df = channel_dfs[0]
        if len(channel_dfs) > 1:
            for ch_df in channel_dfs[1:]:
                df = pd.merge(df, ch_df, left_on=df.index, right_on=ch_df.index, how='outer')
                df.index = df['key_0'].values
                df = df.drop(columns=['key_0'])
        return df

    def _assemble_channels(self, channels: List[str], ignore_channels: List[str] = [], ch_type: Optional[DWChannelType] = None) -> List[str]:
        """Filter channels according to optional criteria"""
        if not channels:
            # Return dataframe of all channels by default
            channels = list(self.keys())

        channels = [ch for ch in channels if ch not in ignore_channels]

        if ch_type is not None:
            channels = [ch for ch in channels if self[ch].channel_type == ch_type]

        return channels

    def dataframe(self, channels: List[str] = [], ignore_channels: List[str] = []) -> pd.DataFrame:
        channels = self._assemble_channels(channels, ignore_channels)
        return self._build_dataframe(channels)

    def sync_dataframe(self, channels: List[str] = [], ignore_channels: List[str] = []) -> pd.DataFrame:
        channels = self._assemble_channels(channels, ignore_channels, DWChannelType.DW_CH_TYPE_SYNC)
        return self._build_dataframe(channels)

    def async_dataframe(self, channels: List[str] = [], ignore_channels: List[str] = []) -> pd.DataFrame:
        channels = self._assemble_channels(channels, ignore_channels, DWChannelType.DW_CH_TYPE_ASYNC)
        return self._build_dataframe(channels)

    def close(self):
        """Close the d7d file and destroy the reader_handle"""
        if not self.closed:
            self.closed = True
            self.clear()  # Delete channel metadata
        if self.reader_handle.value is not None:
            DLL.DWICloseDataFile(self.reader_handle)
            DLL.DWIDestroyReader(self.reader_handle)
            self.reader_handle.value = None

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
    epoch = datetime(1899, 12, 30, tzinfo=timezone.utc)

    def __init__(self):
        super().__init__()

    def __str__(self):
        return f"{self.sample_rate} Hz | {self.start_measure_time} | {self.start_store_time} | {self.duration} s"

    @property
    def start_store_time(self):
        """Return start_store_time in Python datetime format"""
        return self.epoch + timedelta(self._start_store_time)

    @property
    def start_measure_time(self):
        """Return start_store_time in Python datetime format"""
        return self.epoch + timedelta(self._start_measure_time)


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

def loadDLL(dllPath: Optional[Path] = None) -> ctypes.CDLL:
    global DLL
    if not dllPath:
        # Determine appropriate library to load
        dllName = "DWDataReaderLib"
        if platform.architecture()[0] == '64bit':
            dllName += "64"
        dllPath = Path(__file__).with_name(dllName)
        if platform.system() == 'Linux':
            dllPath = dllPath.with_suffix(".so")
        elif platform.system() == 'Darwin':
            dllPath = dllPath.with_suffix(".dylib")
    loader = ctypes.cdll if platform.system() != "Windows" else ctypes.windll # type: ignore[attr-defined]
    DLL = loader[str(dllPath)]
    return DLL

def get_version():
    ver_major = ctypes.c_int()
    ver_minor = ctypes.c_int()
    ver_patch = ctypes.c_int()
    DLL.DWGetVersionEx(ctypes.byref(ver_major), ctypes.byref(ver_minor), ctypes.byref(ver_patch))
    return f"{ver_major.value}.{ver_minor.value}.{ver_patch.value}"

def open_file(source: str) -> DWFile:
    """Open and return a DWFile based on the source filename
    Deprecated: Use DWFile(source) directly instead.
    """
    return DWFile(source)

def decode_bytes(byte_string):
    """Convert bytes to string with proper decoding."""
    if isinstance(byte_string, bytes):
        return byte_string.decode(encoding=encoding, errors='replace').rstrip('\x00')
    return byte_string

DLL = loadDLL()  # initialize the DLL immediately

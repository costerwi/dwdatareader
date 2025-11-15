#!/usr/bin/env python
"""Tests for dwdatareader python module.

Execute with:
    python test_module.py
"""
import os
import datetime
import unittest
import xml.etree.ElementTree as ET

import dwdatareader as dw
import numpy as np
import pandas as pd


class TestDW(unittest.TestCase):
    def setUp(self):
        """ Identify path to the d7d test data file """
        import os
        self.d7dname = os.path.join(os.path.dirname(__file__),
                "Example_Drive01.d7d")
        self.complex_dxd_name = os.path.join(os.path.dirname(__file__),
                "example_complex.dxd")

    def test_context(self):
        """Check that the d7d is open and closed according to context."""
        with dw.DWFile(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            self.assertEqual(d7d.name, self.d7dname)
            ch = d7d['ENG_RPM']

        # The d7d should be closed outside the above context
        self.assertTrue(d7d.closed, 'd7d did not close')
        with self.assertRaises(dw.DWError,
                msg="accessing channel data"):
            ch.series() # ScaledSamplesCount returns -1
        with self.assertRaises(KeyError,
                msg="accessing channel metadata"):
            d7d['ENG_RPM']  # I/O operation on closed file

    def test_DWError(self):
        """Should raise DWError with status and message members"""
        try:
            with dw.DWFile(__file__):
                pass
        except dw.DWError as e:
            self.assertEqual(e.status, dw.DWStatus.DWSTAT_ERROR_FILE_CORRUPT)
            self.assertEqual(e.message, "File is corrupted or has invalid format")

    def test_missing_file(self):
        """Should fail to open missing file"""
        with self.assertRaises(dw.DWError):
            with dw.DWFile("abcdef"):
                pass

    def test_corrupt_file(self):
        """Should fail to open file of wrong format"""
        with self.assertRaises(dw.DWError):
            with dw.DWFile(__file__):
                pass

    def test_keys(self):
        """Check iteration of channel names"""
        with dw.DWFile(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            self.assertEqual(len(d7d), 20)
            for key in d7d:
                self.assertTrue(key.startswith(d7d[key].long_name))

    def test_items(self):
        """Check iteration of items"""
        with dw.DWFile(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            for key, value in d7d.items():
                self.assertTrue(key.startswith(value.long_name))

    def test_alternate_key(self):
        """Check that alternate channel keys are working."""
        with dw.DWFile(self.d7dname, key=lambda ch: ch.index) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            ch = d7d[10]  # using channel.index as key
            self.assertEqual(ch.name, "V_SPEED")
            with self.assertRaises(KeyError, msg="Referring to channel name"):
                d7d['V_SPEED']
        with self.assertRaises(AssertionError, msg="key not callable"):
            dw.DWFile(self.d7dname, key=7)

    def test_info(self):
        """Check that the file info was read correctly."""
        with dw.DWFile(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            self.assertEqual(100, d7d.info.sample_rate)
            start_store_time = d7d.info.start_store_time
            self.assertEqual(21, start_store_time.hour)
            self.assertEqual(9, start_store_time.day)
            self.assertEqual(10, start_store_time.month)
            self.assertEqual(2003, start_store_time.year)
            start_measure_time = d7d.info.start_measure_time
            self.assertEqual(21, start_measure_time.hour)
            self.assertEqual(9, start_measure_time.day)
            self.assertEqual(10, start_measure_time.month)
            self.assertEqual(2003, start_measure_time.year)

    def test_metadata(self):
        """Make sure channel metadata is correctly loaded."""
        with dw.DWFile(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            self.assertEqual(20, len(d7d))
            self.assertEqual('rpm', d7d['ENG_RPM'].unit)

    def test_header(self):
        """Make sure file headers are available."""
        with dw.DWFile(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            # Unfortunately, no headers in sample file
            self.assertDictEqual({}, d7d.header)

    def test_export_header(self):
        """Make sure header is exported to file local.xml"""
        file_name = "local.xml"
        with dw.DWFile(self.d7dname) as dwf:
            dwf.export_header(file_name)
        assert os.path.isfile("local.xml")

    def test_events(self):
        """Make sure events are readable."""
        with dw.DWFile(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            self.assertEqual(2, len(d7d.events()))

    def test_series(self):
        """Read a series and make sure its value matches expectation."""
        with dw.DWFile(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            s = d7d['GPSvel'].series()
            s5 = s[5.0:5.5] # time-based slice!
            self.assertEqual(76.46, round(s5.mean(), 2))

    def test_series_generator(self):
        """Read a series and make sure its value matches expectation."""
        with dw.DWFile(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            channel = d7d['GPSvel']
            expected = channel.series()
            actual = pd.concat(list(channel.series_generator(500)))
            self.assertEqual(len(expected), len(actual))
            self.assertTrue(abs(actual.sum() - expected.sum()) < 1)

    def test_channel_type(self):
        """Channel type"""
        with dw.DWFile(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            channel = d7d['ENG_RPM']
            actual = channel.channel_type
            expected = 1
            self.assertEqual(actual, expected)

    def test_channel_scale(self):
        """Channel Scale"""
        with dw.DWFile(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            channel = d7d['ENG_RPM']
            actual = channel.scale
            expected = 0.25
            self.assertEqual(actual, expected)

    def test_channel_offset(self):
        """Channel Offset"""
        with dw.DWFile(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            channel = d7d['ENG_RPM']
            actual = channel.offset
            expected = 0.0
            self.assertEqual(actual, expected)


    @unittest.expectedFailure
    def test_CAN_channel(self):
        """Read channel data with CAN in its channel_index"""
        with dw.DWFile(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            # Following did not fail prior to DWDataReader v4.2.0.31
            # Now all channels whose channel_index begins with CAN will fail with "Feature or operation not supported on CAN channel"
            d7d['ENG_RPM'].scaled()

    def test_channel_index(self):
        """Channel type"""
        with dw.DWFile(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            channel = d7d['GPSvel']
            actual = channel.channel_index
            expected = 'AI;4'
            self.assertEqual(actual, expected)

    def test_channel_xml(self):
        """Channel type"""
        with dw.DWFile(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            channel = d7d['ENG_RPM']
            xml = channel.channel_xml
            root = ET.fromstring(xml)
            actual = root.find('ForceSinglePrecision').text
            expected = 'True'
            self.assertEqual(actual, expected)

    def test_channel_longname(self):
        """Channel long name"""
        with dw.DWFile(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            channel = d7d['ENG_RPM']
            actual = channel.long_name
            expected = 'ENG_RPM'
            self.assertEqual(actual, expected)

    def test_reduced(self):
        """Read reduced channel data and check value."""
        with dw.DWFile(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            r = d7d['GPSvel'].reduced()
            r5 = r.ave.asof(5.0) # index into reduced list near time=5.0
            self.assertEqual(76.46, round(r5, 2))

    def test_channel_dataframe(self):
        """Read one channel as a DataFrame"""
        with dw.DWFile(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            df = d7d['GPSvel'].dataframe()
            self.assertEqual(len(df.GPSvel), 9580)

    def test_all_dataframe(self):
        """Read all channel data as a single DataFrame."""
        with dw.DWFile(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            channels = [ch_name for ch_name, channel in d7d.items() if not channel.channel_index.startswith('CAN')]
            self.assertEqual((11568, 8), d7d.dataframe(channels).shape)

    def test_sync_dataframe(self):
        """Read all channel data as a single DataFrame."""
        with dw.DWFile(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            channels = [ch_name for ch_name, channel in d7d.items() if not channel.channel_index.startswith('CAN')]
            self.assertEqual((9580, 3), d7d.sync_dataframe(channels).shape)

    def test_async_dataframe(self):
        """Read all channel data as a single DataFrame."""
        with dw.DWFile(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            channels = [ch_name for ch_name, channel in d7d.items() if not channel.channel_index.startswith('CAN')]
            self.assertEqual((1988, 5), d7d.async_dataframe(channels).shape)

    def test_encoding_utf8(self):
        """ Check that encoding is set correcly """
        dw.encoding = 'utf-8'
        with dw.DWFile(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')

    def test_encoding_utf32(self):
        """ Check that wrong encoding raises an error """
        dw.encoding = 'utf-32'
        with self.assertRaises(dw.DWError):
            dw.DWFile(self.d7dname)

    # TODO: need example file and test methods for binary channels

    def test_complex_channel_dtypes(self):
        """Check complex channel dtypes"""
        with dw.DWFile(self.complex_dxd_name) as test_file:
            self.assertFalse(test_file.closed, "dxd did not open")
            complex_channels = [
                channel_name
                for channel_name, channel in test_file.items()
                if isinstance(channel, dw.DWComplexChannel)
            ]
            np.testing.assert_array_equal(
                test_file.dataframe(complex_channels).dtypes.values,
                np.array([np.dtype("complex128")] * len(complex_channels))
            )

    def test_complex_channel_shape(self):
        """Check complex channel shape"""
        with dw.DWFile(self.complex_dxd_name) as test_file:
            complex_channels = [
                channel_name
                for channel_name, channel in test_file.items()
                if isinstance(channel, dw.DWComplexChannel)
            ]
            self.assertEqual(len(complex_channels), 33)
            self.assertEqual(test_file.dataframe(complex_channels).shape, (184, 33))


if __name__ == '__main__':
    unittest.main()

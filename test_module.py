#!/usr/bin/env python
"""Tests for dwdatareader python module.

Execute with:
    python test_module.py
"""
import os
import unittest
import xml.etree.ElementTree as ET
import pandas as pd
import dwdatareader as dw


class TestDW(unittest.TestCase):
    def setUp(self):
        """ Identify path to the d7d test data file """
        import os
        self.d7dname = os.path.join(os.path.dirname(__file__),
                "Example_Drive01.d7d")

    def test_context(self):
        """Check that the d7d is open and closed according to context."""
        with dw.open_file(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            ch = d7d['ENG_RPM']

        # The d7d should be closed outside the above context
        self.assertTrue(d7d.closed, 'd7d did not close')
        with self.assertRaises(ValueError,
                msg="accessing channel data"):
            ch.series() # ScaledSamplesCount returns -1
        with self.assertRaises(KeyError,
                msg="accessing channel metadata"):
            d7d['ENG_RPM']  # I/O operation on closed file

    def test_keys(self):
        """Check iteration of channel names"""
        with dw.open_file(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            for key in d7d:
                self.assertEqual(key, d7d[key].name)

    def test_items(self):
        """Check iteration of items"""
        with dw.open_file(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            for key, value in d7d.items():
                self.assertEqual(key, value.name)

    def test_info(self):
        """Check that the file info was read correctly."""
        with dw.open_file(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            self.assertEqual(100, d7d.info.sample_rate)

    def test_filelike(self):
        """Test module access and usage of filelike objects."""
        with open(self.d7dname, 'rb') as f:
            self.assertFalse(f.closed, 'file did not open')

    def test_metadata(self):
        """Make sure channel metadata is correctly loaded."""
        with dw.open_file(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            self.assertEqual(20, len(d7d))
            self.assertEqual('rpm', d7d['ENG_RPM'].unit)

    def test_header(self):
        """Make sure file headers are available."""
        with dw.open_file(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            # Unfortunately, no headers in sample file
            self.assertDictEqual({}, d7d.header)

    def test_export_header(self):
        """Make sure header is exported to file local.xml"""
        file_name = "local.xml"
        with dw.open_file(self.d7dname) as dwf:
            dwf.export_header(file_name)
        assert os.path.isfile("local.xml")

    def test_events(self):
        """Make sure events are readable."""
        with dw.open_file(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            self.assertEqual(2, len(d7d.events()))

    def test_series(self):
        """Read a series and make sure its value matches expectation."""
        with dw.open_file(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            s = d7d['GPSvel'].series()
            s5 = s[5.0:5.5] # time-based slice!
            self.assertEqual(76.46, round(s5.mean(), 2))

    def test_series_generator(self):
        """Read a series and make sure its value matches expectation."""
        with dw.open_file(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            channel = d7d['GPSvel']
            expected = channel.series()
            actual = pd.concat(list(channel.series_generator(500)))
            self.assertEqual(len(expected), len(actual))
            self.assertTrue(abs(actual.sum() - expected.sum()) < 1)

    def test_channel_type(self):
        """Channel type"""
        with dw.open_file(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            channel = d7d['ENG_RPM']
            actual = channel.channel_type
            expected = 1
            self.assertEqual(actual, expected)

    def test_channel_scale(self):
        """Channel Scale"""
        with dw.open_file(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            channel = d7d['ENG_RPM']
            actual = channel.scale
            expected = 0.25
            self.assertEqual(actual, expected)

    def test_channel_offset(self):
        """Channel Offset"""
        with dw.open_file(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            channel = d7d['ENG_RPM']
            actual = channel.offset
            expected = 0.0
            self.assertEqual(actual, expected)


    @unittest.expectedFailure
    def test_CAN_channel(self):
        """Read channel data with CAN in its channel_index"""
        with dw.open_file(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            # Following did not fail prior to DWDataReader v4.2.0.31
            # Now all channels whose channel_index begins with CAN will fail with "Feature or operation not supported on CAN channel"
            scaled = d7d['ENG_RPM'].scaled()

    def test_channel_index(self):
        """Channel type"""
        with dw.open_file(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            channel = d7d['GPSvel']
            actual = channel.channel_index
            expected = 'AI;4'
            self.assertEqual(actual, expected)

    def test_channel_xml(self):
        """Channel type"""
        with dw.open_file(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            channel = d7d['ENG_RPM']
            xml = channel.channel_xml
            root = ET.fromstring(xml)
            actual = root.find('ForceSinglePrecision').text
            expected = 'True'
            self.assertEqual(actual, expected)

    def test_channel_longname(self):
        """Channel long name"""
        with dw.open_file(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            channel = d7d['ENG_RPM']
            actual = channel.long_name
            expected = 'ENG_RPM'
            self.assertEqual(actual, expected)

    def test_reduced(self):
        """Read reduced channel data and check value."""
        with dw.open_file(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            r = d7d['GPSvel'].reduced()
            r5 = r.ave.asof(5.0) # index into reduced list near time=5.0
            self.assertEqual(76.46, round(r5, 2))

    def test_channel_dataframe(self):
        """Read one channel as a DataFrame"""
        with dw.open_file(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            df = d7d['GPSvel'].dataframe()
            self.assertEqual(len(df.GPSvel), 9580)

    def test_all_dataframe(self):
        """Read all channel data as a single DataFrame."""
        with dw.open_file(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            channels = [ch_name for ch_name, channel in d7d.items() if not channel.channel_index.startswith('CAN')]
            self.assertEqual((11568, 8), d7d.dataframe(channels).shape)

    def test_sync_dataframe(self):
        """Read all channel data as a single DataFrame."""
        with dw.open_file(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            channels = [ch_name for ch_name, channel in d7d.items() if not channel.channel_index.startswith('CAN')]
            self.assertEqual((9580, 3), d7d.sync_dataframe(channels).shape)

    def test_async_dataframe(self):
        """Read all channel data as a single DataFrame."""
        with dw.open_file(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            channels = [ch_name for ch_name, channel in d7d.items() if not channel.channel_index.startswith('CAN')]
            self.assertEqual((1988, 5), d7d.async_dataframe(channels).shape)

    def test_encoding_utf8(self):
        """ Check that encoding is set correcly """
        dw.encoding = 'utf-8'
        with dw.open_file(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')

    def test_encoding_utf32(self):
        """ Check that wrong encoding raises an error """
        dw.encoding = 'utf-32'
        with self.assertRaises(RuntimeError):
            dw.open_file(self.d7dname)


if __name__ == '__main__':
    unittest.main()

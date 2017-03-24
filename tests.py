#!/usr/bin/env python
"""Tests for dwdatareader python module.

Execute with:
    python tests.py
"""

import unittest

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
        with dw.open(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            ch = d7d['ENG_RPM']

        # The d7d should be closed outside of the above context
        self.assertTrue(d7d.closed, 'd7d did not close')
        with self.assertRaises(IndexError,
                msg="accessing channel data"):
            ch.series() # ScaledSamplesCount returns -1
        with self.assertRaises(ValueError,
                msg="accessing channel metadata"):
            d7d['ENG_RPM'] # I/O operation on closed file

    def test_info(self):
        """Check that the file info was read correctly."""
        with dw.open(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            self.assertEqual(100, d7d.info.sample_rate)

    def test_filelike(self):
        """Test module access and usage of filelike objects.

        In this test the dw module must create a temporary file copy
        of the d7d filelike object so that the dll will have something to
        open.  The temporary d7d file should be removed when the DWFile is
        closed."""

        import os.path
        with open(self.d7dname, 'rb') as f:
            self.assertFalse(f.closed, 'file did not open')
            with dw.open(f) as d7d:
                self.assertFalse(d7d.closed, 'temporary d7d did not open')
            self.assertFalse(os.path.exists(d7d.name),
                    'temporary d7d not removed')

    def test_metadata(self):
        """Make sure channel metadata is correctly loaded."""
        with dw.open(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            self.assertEqual(20, len(d7d))
            self.assertEqual('rpm', d7d['ENG_RPM'].unit)

    def test_header(self):
        """Make sure file headers are available."""
        with dw.open(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            # Unfortunately, no headers in sample file
            self.assertDictEqual({}, d7d.header)

    def test_events(self):
        """Make sure events are readable."""
        with dw.open(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            self.assertEqual(2, len(d7d.events()))

    def test_series(self):
        """Read a series and make sure its value matches expectation."""
        with dw.open(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            s = d7d['ENG_RPM'].series()
            s5 = s[5.0:5.5] # time-based slice!
            self.assertTrue(abs(s5.mean() - 3098.5) < 1)

    def test_series_generator(self):
        """Read a series and make sure its value matches expectation."""
        with dw.open(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            channel = d7d['ENG_RPM']
            nos = channel.number_of_samples
            expected = channel.series()
            actual = pd.concat(list(channel.series_generator(500)))
            actual = actual.iloc[:nos]
            self.assertTrue(actual.equals(expected))

    def test_reduced(self):
        """Read reduced channel data and check value."""
        with dw.open(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            r = d7d['ENG_RPM'].reduced()
            r5 = r.ave.asof(5.0) # index into reduced list near time=5.0
            self.assertTrue(abs(r5 - 3099.7) < 1)

    def test_dataframe(self):
        """Read all channel data as a single DataFrame."""
        with dw.open(self.d7dname) as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')
            self.assertEqual((18771, 20), d7d.dataframe().shape)

    def test_encoding_uft8(self):
        """ Check that encoding is set correcly """
        with dw.open(self.d7dname, encoding="utf-8") as d7d:
            self.assertFalse(d7d.closed, 'd7d did not open')

    def test_encoding_uft32(self):
        """ Check that wrong encoding raises an error """
        with self.assertRaises(dw.DWError):
            dw.open(self.d7dname, encoding="utf-32")

if __name__ == '__main__':
    unittest.main()

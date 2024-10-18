"""Tests for the estimation components of pylord"""

import unittest
import pandas as pd
import numpy as np
from pylord.estimation.cutoff_finder import PeakDipFinder, MainDipCutoff, FixedCutoff


class TestPeakDipFinder(unittest.TestCase):


    def test_find_peaks_and_dips_no_dips(self):
        kde = np.array([1, 2, 3, 4, 5])
        axes = np.arange(len(kde))
        result = PeakDipFinder.find_peaks(axes, kde)
        self.assertEqual(len(result), 0)

    def test_find_peaks_and_dips_one_dip(self):
        kde = np.array([1, 2, 1, 4, 5])
        axes = np.arange(len(kde))
        result = PeakDipFinder.find_peaks(axes, kde)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], 2)

    def test_find_peaks_and_dips_multiple_dips(self):
        kde = np.array([1, 2, 1, 4, 3, 4, 2])
        axes = np.arange(len(kde))
        result = PeakDipFinder.find_peaks(axes, kde)
        self.assertEqual(len(result), 2)
        self.assertTrue(2 in result)
        self.assertTrue(4 in result)

class TestMainDipCutoff(unittest.TestCase):

    def test_find_cutoff_empty_dataframe(self):
        df = pd.DataFrame({'score': []})
        main_dip_cutoff = MainDipCutoff(df, 'score')
        with self.assertRaises(ValueError):
            main_dip_cutoff.find_cutoff()

    def test_find_cutoff_no_dips(self):
        df = pd.DataFrame({'score': [1, 2, 3, 4, 5]})
        main_dip_cutoff = MainDipCutoff(df, 'score')
        self.assertEqual(main_dip_cutoff.find_cutoff(), np.median(df['score'].values))

    def test_find_cutoff_with_dips(self):
        df = pd.DataFrame({'score': [1, 2, 1, 4, 3, 4, 2]})
        main_dip_cutoff = MainDipCutoff(df, 'score')
        expected_dip = 2
        self.assertEqual(main_dip_cutoff.find_cutoff(), expected_dip)

class TestFixedCutoff(unittest.TestCase):

    def test_find_cutoff(self):
        df = pd.DataFrame({'score': [1, 2, 3]})
        fixed_cutoff = FixedCutoff(df, 'score')
        self.assertEqual(fixed_cutoff.find_cutoff(), 0.23)


if __name__ == '__main__':
    unittest.main()
import unittest
import pandas as pd
import numpy as np
import sys
import os
from pprint import pprint
import json
from scripts.utils import DataLoader
from scripts.cleaning import CleanDataFrame

sys.path.append('../')

class TestCleanDataFrame(unittest.TestCase):

    def setUp(self) -> pd.DataFrame:
        self.test_df = DataLoader("tests", "test_data.csv").read_csv()
        self.cleaner = CleanDataFrame()

    def test_remove_null_row(self):
        df = self.cleaner.remove_null_row(self.test_df, self.test_df.columns)
        self.assertEqual(len(df), 3)


if __name__ == '__main__':
    unittest.main()

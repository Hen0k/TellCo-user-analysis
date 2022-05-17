import unittest
import pandas as pd
import numpy as np
import sys
import os
from pprint import pprint
import json
from scripts.utils import DataLoader
from scripts.exploration import Analysis

sys.path.append('../')


class TestAnalysis(unittest.TestCase):

    def setUp(self) -> pd.DataFrame:
        self.test_df = DataLoader("tests", "test_data.csv").read_csv()
        self.cleaner = Analysis()


if __name__ == '__main__':
    unittest.main()


import unittest
import numpy as np
from data_assimilation_package import DataAssimilation

class TestDataAssimilation(unittest.TestCase):
    def test_assimilation(self):
        gridded_data = np.random.rand(256, 256)
        station_data = np.random.rand(256, 256)
        assimilator = DataAssimilation()
        result = assimilator.assimilate(gridded_data, station_data)
        self.assertIsNotNone(result)

if __name__ == "__main__":
    unittest.main()

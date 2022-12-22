import unittest

import prediction.prediction as prediction
import vegetation.vegetation as vegetation


class TestPediction(unittest.TestCase):
    def test_teachModel(self):
        arrays = vegetation.convertImgsToArrays('bolivia')
        self.assertEqual(prediction.teachModel(arrays, 'bolivia'), True)

    def test_predictFutureVegetation(self):
        arrays = vegetation.convertImgsToArrays('bolivia')
        self.assertEqual(prediction.predictFutureVegetation(arrays[0], 'bolivia'), True)




if __name__ == '__main__':
    unittest.main()
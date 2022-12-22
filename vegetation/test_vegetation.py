import unittest
import vegetation


class TestVegetation(unittest.TestCase):
    def test_getvegetation(self):
        self.assertEqual(vegetation.getvegetation("bolivia"), True)
        self.assertEqual(vegetation.getvegetation("china"), True)

    def test_createoverlay(self):
        self.assertRaises(FileNotFoundError, vegetation.createOverlay, "france")
        self.assertEqual(vegetation.createOverlay("bolivia"), True)
        self.assertEqual(vegetation.createOverlay("china"), True)



if __name__ == '__main__':
    unittest.main()
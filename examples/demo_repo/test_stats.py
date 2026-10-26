import unittest

from stats import mean


class MeanTest(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(mean([2, 4, 6]), 4)

    def test_single_value(self):
        self.assertEqual(mean([5]), 5)


if __name__ == "__main__":
    unittest.main()

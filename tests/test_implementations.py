import unittest
import numpy as np
import src.implementations as imp


class TestImplementations(unittest.TestCase):
    # Naive Test
    # Test case 1
    def test_naive_python_all_ones_case(self):
        A = [[1, 1], [1, 1]]
        B = [[1, 1], [1, 1]]        
        expected = [[2, 2], [2, 2]]
        result = imp.naive_python_implementation(A, B)
        self.assertTrue(np.array_equal(result, expected))

    # Test case 2
    def test_naive_rectangular_case(self):
        A = [[1, 2, 3], [4, 5, 6]]
        B = [[7, 8], [9, 10], [11, 12]]
        expected = [[58, 64], [139, 154]]
        result = imp.naive_python_implementation(A, B)
        self.assertTrue(np.array_equal(result, expected))
        
    # Test case 3
    def test_naive_identity_case(self):
        A = [[1, 0], [0, 1]]
        B = [[5, 6], [7, 8]]
        expected = [[5, 6], [7, 8]]
        result = imp.naive_python_implementation(A, B)
        self.assertTrue(np.array_equal(result, expected))

    # Test case 4
    def test_naive_zero_case(self):
        A = [[0, 0], [0, 0]]
        B = [[1, 2], [3, 4]]
        expected = [[0, 0], [0, 0]]
        result = imp.naive_python_implementation(A, B)
        self.assertTrue(np.array_equal(result, expected))
        
    # Numpy test
    # Test 1
    def test_numpy_all_ones_case(self):
        A = [[1, 1], [1, 1]]
        B = [[1, 1], [1, 1]]
        expected = [[2, 2], [2, 2]]
        result = imp.numpy_implementation(A, B)
        self.assertTrue(np.array_equal(result, expected))
        
    # Test case 2
    def test_numpy_rectangular_case(self):
        A = [[1, 2, 3], [4, 5, 6]]
        B = [[7, 8], [9, 10], [11, 12]]
        expected = [[58, 64], [139, 154]]
        result = imp.numpy_implementation(A, B)
        self.assertTrue(np.array_equal(result, expected))
        
    # Test case 3
    def test_numpy_identity_case(self):
        A = [[1, 0], [0, 1]]
        B = [[5, 6], [7, 8]]
        expected = [[5, 6], [7, 8]]
        result = imp.numpy_implementation(A, B)
        self.assertTrue(np.array_equal(result, expected))

    # Test case 4
    def test_numpy_zero_case(self):
        A = [[0, 0], [0, 0]]
        B = [[1, 2], [3, 4]]
        expected = [[0, 0], [0, 0]]
        result = imp.numpy_implementation(A, B)
        self.assertTrue(np.array_equal(result, expected))

if __name__ == "__main__":
    unittest.main()

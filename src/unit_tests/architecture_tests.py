import unittest
import numpy as np
import collections
import neural_logic_machines.architecture as architecture

class TestArchitecture(unittest.TestCase):

    arch = architecture.Architecture()

    def test_permute_unary(self):
        # Given
        input = np.array([1, 0, 0, 0])

        # When
        output = self.arch.permute(input)

        # Then
        self.assertTrue((output == np.array([[1, 0, 0, 0]])).all())

    def test_permute_binary(self):
        # Given
        input = np.array([[0, 0, 1, 0],
                          [0, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, 1]])
        
        # When
        output = self.arch.permute(input)

        # Then
        self.assertTrue(
            (output ==
             np.array([[[0, 0, 1, 0],
                        [0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]],
                       [[0, 0, 0, 0],
                        [0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1]]])).all())
        
    def test_select_body_2_unary_predicates(self):
        # Given
        input = np.array([[0, 0, 1, 0],
                          [0, 1, 0, 0]])

        # When
        output = self.arch.select_body(input)

        # Then
        self.assertTrue(self.same_np_arrays(output, [
            np.array([[]]),
            np.array([[0, 0, 1, 0]]),
            np.array([[0, 1, 0, 0]]),
            np.array([[0, 0, 1, 0],
                      [0, 1, 0, 0]])]))

    def test_select_body_3_binary_predicates(self):
        # Given
        input = np.array([[[0, 0], [0, 0]],
                          [[0, 1], [1, 0]],
                          [[1, 1], [1, 1]]])
        
        # When
        output = self.arch.select_body(input)

        # Then
        self.assertTrue(self.same_np_arrays(output, [
            np.array([[[]]]),
            np.array([[[0, 0], [0, 0]]]),
            np.array([[[0, 1], [1, 0]]]),
            np.array([[[1, 1], [1, 1]]]),
            np.array([[[0, 0], [0, 0]],
                      [[0, 1], [1, 0]]]),
            np.array([[[0, 0], [0, 0]],
                      [[1, 1], [1, 1]]]),
            np.array([[[0, 1], [1, 0]],
                      [[1, 1], [1, 1]]]),
            np.array([[[0, 0], [0, 0]],
                      [[0, 1], [1, 0]],
                      [[1, 1], [1, 1]]])]))

    def same_np_arrays(self, x, y):
        if len(x) != len(y):
            return False
        
        return all([any([np.array_equiv(a1, a2) for a2 in y]) for a1 in x])

if __name__ == '__main__':
    unittest.main()
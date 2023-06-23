import unittest
import numpy as np
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

if __name__ == '__main__':
    unittest.main()
import unittest
import numpy as np
import neural_logic_machines.architecture as architecture

class TestArchitecture(unittest.TestCase):

    sut = architecture.Architecture()

    def test_permute_unary(self):
        # Given
        predicates = np.array([[1, 0, 0, 0]])

        # When
        result = self.sut.permute_predicate(predicates)

        # Then
        self.assertTrue((result == np.array([[1, 0, 0, 0]])).all())

    def test_permute_binary(self):
        # Given
        predicates = np.array([[[0, 0, 1, 0],
                           [0, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1]]])
        
        # When
        result = self.sut.permute_predicate(predicates)

        # Then
        self.assertTrue(
            (result ==
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
        predicates = np.array([[0, 0, 1, 0],
                          [0, 1, 0, 0]])

        # When
        result = self.sut.select_body(predicates)

        # Then
        self.assertTrue(self.same_np_arrays(result, [
            np.array([[1, 1, 1, 1]]),
            np.array([[0, 0, 1, 0]]),
            np.array([[0, 1, 0, 0]]),
            np.array([[0, 0, 1, 0],
                      [0, 1, 0, 0]])]))

    def test_select_body_3_binary_predicates(self):
        # Given
        predicates = np.array([[[0, 0], [0, 0]],
                          [[0, 1], [1, 0]],
                          [[1, 1], [1, 1]]])
        
        # When
        result = self.sut.select_body(predicates)

        # Then
        self.assertTrue(self.same_np_arrays(result, [
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
        
    def test_expand(self):
        # Given
        predicates = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
        num_objects = 4

        # When
        result = self.sut.expand(predicates, num_objects)

        # Then
        self.assertTrue(
            (result ==
             np.array([[[0, 1, 0, 1],
                        [0, 1, 0, 1],
                        [0, 1, 0, 1],
                        [0, 1, 0, 1]],
                       [[0, 0, 1, 1],
                        [0, 0, 1, 1],
                        [0, 0, 1, 1],
                        [0, 0, 1, 1]]])).all())
        
    def test_reduce(self):
        # Given
        predicates = np.array([[[1, 0, 1, 0],
                                [0, 0, 1, 1],
                                [0, 0, 1, 0],
                                [0, 0, 1, 1]],
                               [[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 0]]])

        # When
        result = self.sut.reduce(predicates)

        # Then
        self.assertTrue((result ==
             np.array([[1, 0, 1, 1], 
                       [0, 1, 0, 0]])).all())
        
    def test_bool_logic_simple(self):
        # Given
        premise = [np.array([0]),
                   np.array([[1, 1, 1]]),
                   np.array([[[0, 1, 0],
                              [0, 0, 1],
                              [0, 0, 0]],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]])]
        
        weights = [np.array([0]*4),
                   np.array([0]*16),
                   np.array([0, 0, 0, 0, 1] + [0]*59 + [0]*64)]

        # When
        result = self.sut.apply(weights, premise)

        # Then
        self.assertTrue((result[0] == np.array([0])).all())
        self.assertTrue((result[1] == np.array([[1, 1, 1]])).all())
        self.assertTrue((result[2] == np.array([[[0, 1, 0],
                                                 [1, 0, 1],
                                                 [0, 1, 0]],
                                                [[0, 0, 0],
                                                 [0, 0, 0],
                                                 [0, 0, 0]]])).all())
    
    def test_bool_logic_complex(self):
        # Given
        premise = [np.array([]),
                   np.array([[0, 0, 0, 1],
                             [0, 0, 1, 0]]),
                   np.array([[[0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 1],
                              [0, 0, 1, 0]],
                             [[0, 1, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]],
                             [[0, 0, 0, 0],
                              [1, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]]])]

        weights = np.array([[], [0]*64, [0]*3072], dtype=object)
        weights[1][4] = 1
        weights[1][37] = 1
        weights[2][1038] = 1
        weights[2][2077] = 1

        # When
        result = self.sut.apply(weights, premise)

        # Then
        self.assertTrue((result[0] == np.array([])).all())
        self.assertTrue((result[1] == np.array([[0, 1, 0, 1],
                                                [1, 0, 1, 0]])).all())
        self.assertTrue((result[2] == np.array([[[0, 0, 0, 0],
                                                 [0, 0, 0, 0],
                                                 [0, 0, 0, 1],
                                                 [0, 0, 1, 0]],
                                                [[0, 1, 0, 0],
                                                 [0, 0, 0, 0],
                                                 [0, 0, 0, 1],
                                                 [0, 0, 0, 0]],
                                                [[0, 0, 0, 0],
                                                 [1, 0, 0, 0],
                                                 [0, 0, 0, 0],
                                                 [0, 0, 1, 0]]])).all())

    def same_np_arrays(self, x, y):
        if len(x) != len(y):
            return False
        
        return all([any([np.array_equiv(a1, a2) for a2 in y]) for a1 in x])

if __name__ == '__main__':
    unittest.main()
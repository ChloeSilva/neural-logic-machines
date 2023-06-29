import unittest
import neural_logic_machines.problem as prob
import numpy as np

class TestProblem(unittest.TestCase):

    def test_process(self):
        # Given
        data = ['in:', 'edge(a, b).', '',
                'out:', 'edge(a, b).', 'edge(b, a).', '',
                'in:', 'edge(c, d).', '',
                'out:', 'edge(c, d).', 'edge(d, c).', '',]

        problem = prob.Problem(
            max_predicates = [0, 1, 2],
            max_body = 3,
            predicate_names = [[], 
                               ['node'],
                               ['edge', 'connected']],
            knowledge_base = [],
            objects = ['a', 'b', 'c', 'd']
        )

        # When
        result = problem.process(data)

        # Then
        self.assertEqual(result, 
            [[['edge(a, b).'], ['edge(a, b).', 'edge(b, a).']],
             [['edge(c, d).'], ['edge(c, d).', 'edge(d, c).']]])

    def test_train_simple(self):
        # Given
        problem = prob.Problem(
            max_predicates = [0, 0, 1],
            max_body = 2,
            predicate_names = [[], 
                               [],
                               ['edge']],
            knowledge_base = [],
            objects = ['a', 'b', 'c', 'd']
        )

        learning_rate = 0.1
        iterations = 100
        
        # When
        result = problem.train('data/train_unit_test_1.txt', learning_rate, iterations)

        # Then
        self.assertEqual(round(result[2][0]), 0)
        self.assertEqual(round(result[2][2]), 1)

    def test_run(self):
        # Given
        problem = prob.Problem(
            max_predicates = [0, 2, 3],
            max_body = 11,
            predicate_names = [[], 
                               ['male', 'female'], 
                               ['sibling', 'brother', 'sister']],
            knowledge_base = ['male(X0) :- brother(X0, X1).',
                              'female(X0) :- sister(X0, X1).',
                              'brother(X0, X1) :- male(X0), sibling(X0, X1).',
                              'sister(X0, X1) :- female(X0), sibling(X0, X1).'],
            objects = ['alice', 'bob', 'carol', 'dave']
        )

        weights = np.array([[], [0]*64, [0]*3072], dtype=object)
        weights[1][4] = 1
        weights[1][37] = 1
        weights[2][1038] = 1
        weights[2][2077] = 1

        problem.solution = weights
        threshold = 0.7

        # Then
        result = problem.run('data/input_unit_test_1.txt', threshold)

        # When
        self.assertEqual(result, ['male(bob).',
                                  'male(dave).',
                                  'female(alice).',
                                  'female(carol).',
                                  'sibling(dave, carol).',
                                  'sibling(carol, dave).',
                                  'brother(bob, alice).',
                                  'brother(dave, carol).',
                                  'sister(alice, bob).',
                                  'sister(carol, dave).'])
        
    def test_rules(self):
        # Given
        problem = prob.Problem(
            max_predicates = [0, 2, 3],
            max_body = 11,
            predicate_names = [[], 
                               ['male', 'female'], 
                               ['sibling', 'brother', 'sister']],
            knowledge_base = [],
            objects = ['alice', 'bob', 'carol', 'dave']
        )

        weights = np.array([[], [0]*64, [0]*3072], dtype=object)
        weights[1][4] = 1
        weights[1][37] = 1
        weights[2][1038] = 1
        weights[2][2077] = 1

        problem.solution = weights
        threshold = 0.7

        # Then
        result = problem.rules(threshold)

        # When
        self.assertEqual(result, [
            'male(X0) :- brother(X0, X1).',
            'female(X0) :- sister(X0, X1).',
            'brother(X0, X1) :- male(X0), sibling(X0, X1).',
            'sister(X0, X1) :- female(X0), sibling(X0, X1).'])

if __name__ == '__main__':
    unittest.main()
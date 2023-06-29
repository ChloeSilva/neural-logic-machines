import unittest
import numpy as np
import neural_logic_machines.interpreter as interpreter
import neural_logic_machines.problem as prob

class TestInterpreter(unittest.TestCase):

    sut = interpreter.Interpreter()

    def test_input_to_tensor(self):
        # Given
        problem = prob.Problem(
            max_predicates = [0, 2, 3],
            max_body = 3,
            predicate_names = [[], 
                               ['male', 'female'],
                               ['sibling', 'brother', 'sister']],
            knowledge_base = [],
            objects = ['alice', 'bob', 'carol', 'dave']
        )
        input = ['male(dave).',
                 'female(carol).',
                 'sibling(dave, carol).',
                 'sibling(carol, dave).',
                 'brother(bob, alice).',
                 'sister(alice, bob).']

        # When
        result = self.sut.input_to_tensor(input, problem)

        # Then
        self.assertTrue((result[0] == np.array([])).all())
        self.assertTrue((result[1] == np.array([[0, 0, 0, 1],
                                                [0, 0, 1, 0]])).all())
        self.assertTrue((result[2] == np.array([[[0, 0, 0, 0],
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
                                                 [0, 0, 0, 0]]])).all())

    def test_tensor_to_output(self):
        # Given
        tensor = [np.array([]),
                  np.array([[0, 1, 0, 1],
                            [1, 0, 1, 0]]),
                  np.array([[[0, 0, 0, 0],
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
                             [0, 0, 1, 0]]])]
        
        problem = prob.Problem(
            max_predicates = [0, 2, 3],
            max_body = 5,
            predicate_names = [[], 
                               ['male', 'female'],
                               ['sibling', 'brother', 'sister']],
            knowledge_base = [],
            objects = ['alice', 'bob', 'carol', 'dave']
        )

        threshold = 1

        # When
        result = self.sut.tensor_to_output(tensor, problem, threshold)

        # Then
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

    def test_predicates_to_weights_simple(self):
        # Given
        problem = prob.Problem(
            max_predicates = [0, 0, 2],
            max_body = 5,
            predicate_names = [[], [], ['edge', 'connected']],
            knowledge_base = ['edge(X0, X1) :- edge(X1, X0).'],
            objects = ['a', 'b', 'c']
            )

        weights = np.array([[], [], [0]*32], dtype=object)
        weights[2][2] = 1

        # When
        result = self.sut.predicates_to_weights(problem)

        # Then
        self.assertTrue((result == weights).all())

    def test_predicates_to_weights_complex(self):
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
            objects = ['alice', 'bob', 'carol']
            )
        
        weights = np.array([[], [0]*64, [0]*3072], dtype=object)
        weights[1][4] = 1
        weights[1][37] = 1
        weights[2][1038] = 1
        weights[2][2077] = 1

        # When
        result = self.sut.predicates_to_weights(problem)

        # Then
        self.assertTrue((result == weights).all())
        
    def test_weights_to_predicates_simple(self):
        # Given
        weights = np.array([[], [], [0]*32], dtype=object)
        weights[2][2] = 1

        problem = prob.Problem(
            max_predicates = [0, 0, 2],
            max_body = 5,
            predicate_names = [[], [], ['edge', 'connected']],
            knowledge_base = [],
            objects = ['a', 'b', 'c']
            )

        problem.solution = weights

        threshold = 1

        # When
        result = self.sut.weights_to_predicates(problem, threshold)

        # Then
        self.assertEqual(result, ['edge(X0, X1) :- edge(X1, X0).'])

    def test_weights_to_predicates_complex(self):
        # Given
        weights = [[], [0]*64, [0]*3072]
        weights[1][4] = 1
        weights[1][37] = 1
        weights[2][1038] = 1
        weights[2][2077] = 1

        problem = prob.Problem(
            max_predicates = [0, 2, 3],
            max_body = 11,
            predicate_names = [[], 
                               ['male', 'female'], 
                               ['sibling', 'brother', 'sister']],
            knowledge_base = [],
            objects = ['alice', 'bob', 'carol']
            )
        
        problem.solution = weights
        threshold = 1

        # When
        result = self.sut.weights_to_predicates(problem, threshold)

        # Then
        self.assertEqual(result, [
            'male(X0) :- brother(X0, X1).',
            'female(X0) :- sister(X0, X1).',
            'brother(X0, X1) :- male(X0), sibling(X0, X1).',
            'sister(X0, X1) :- female(X0), sibling(X0, X1).'])

    def test_internal(self):
        # Given
        names = [[], ['node', 'graph'], ['edge', 'connected']]

        # When
        result = self.sut.internal(names)

        # Then
        self.assertEqual(result, 
           [['node', 'graph'], 
            ['node', 'graph', 'edge', 'connected'],
            ['node', 'graph', 'edge', 'connected']])

    def test_clause_index_to_text(self):
        # Given
        arity = 2
        head_predicate = 1
        body_predicates = [(0, 'normal'), (1, 'normal')]
        body_permutations = [(0, 2), (2, 1)]
        predicate_names = ['edge', 'connected']

        # When
        result = self.sut.clause_index_to_text(
            arity,
            head_predicate,
            body_predicates,
            body_permutations,
            predicate_names
        )

        # Then
        self.assertEqual(
            result, 
               'connected(X0, X1) :- edge(X0, X2), connected(X2, X1).'
            )

    def test_perm_to_text(self):
        # Given
        perm = (0, 3, 1, 2)

        # When
        result = self.sut.perm_to_text(perm, "normal")

        # Then
        self.assertEqual(result, '(X0, X3, X1, X2)')

    def test_names_to_index(self):
        # Given
        rule = 'edge'
        names = [[], ['node'], ['edge', 'connected']]

        # When
        result = self.sut.name_to_index(rule, names)

        # Then
        self.assertEqual(result, (2, 0))

    def test_get_arguments(self):
        # Given
        fact = 'brother(dave, alice).'
        objects = ['alice', 'bob', 'carol', 'dave']

        # When
        result = self.sut.get_arguments(fact, objects)

        # Then
        self.assertEqual(result, [3, 0])

    def test_get_head(self):
        # Given
        rule = 'edge(X0, X1) :- edge(X1, X0), connected(X2, X1)'
        names = [[], ['node'], ['edge', 'connected']]

        # When
        result = self.sut.get_head(rule, names)

        # Then
        self.assertEqual(result, (2, 0))

    def test_get_body(self):
        # Given
        rule = 'edge(X0, X1) :- edge(X0, X2), connected(X2, X1)'
        names = [[], ['node'], ['edge', 'connected']]
        layer = 2

        # When
        result = self.sut.get_body(rule, names, layer)

        # Then
        self.assertEqual(result, [((2, 0), (0, 2)), ((2, 1), (2, 1))])

if __name__ == '__main__':
    unittest.main()
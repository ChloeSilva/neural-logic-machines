import unittest
import numpy as np
import neural_logic_machines.interpreter as interpreter
import neural_logic_machines.problem as problem

class TestInterpreter(unittest.TestCase):

	sut = interpreter.Interpreter()
        
	def test_weights_to_predicates(self):
        # Given
		weights = np.array([[], [], [0,0,1] + [0]*29], dtype=object)

		prob = problem.Problem(
			max_predicates = [0, 0, 2],
			predicate_names = [[], [], ['edge', 'connected']]
		)

		threshold = 1

		# When
		result = self.sut.weights_to_predicates(weights, prob, threshold)

		# Then
		self.assertEqual(result, ['edge(X0, X1) :- edge(X1, X0).'])

	def test_clause_index_to_text(self):
		# Given
		arity = 2
		head_predicate = 1
		body_predicates = [0, 1]
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
		result = self.sut.perm_to_text(perm)

		# Then
		self.assertEqual(result, '(X0, X3, X1, X2)')

if __name__ == '__main__':
    unittest.main()
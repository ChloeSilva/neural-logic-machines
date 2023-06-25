import unittest
import neural_logic_machines.problem as problem

class TestInterpreter(unittest.TestCase):

    def test_problem(self):
        # Given
        p = problem.Problem(
            max_predicates = [0, 3, 2],
            predicate_names = [[], ['a', 'b', 'c'], ['x', 'y']]
        )

        # When
        max_predicates = p.max_predicates
        predicate_names = p.predicate_names

        # Then
        self.assertEqual(max_predicates, [0, 3, 2])
        self.assertEqual(predicate_names, [[], ['a', 'b', 'c'], ['x', 'y']])

    def test_solved_problem(self):
        # Given
        p = problem.Problem(
            max_predicates = [0, 3, 2],
            predicate_names = [[], ['a', 'b', 'c'], ['x', 'y']]
        )

        sp = problem.SolvedProblem(
            problem = p,
            solution = [[], [0, 1, 1, 0], [1, 0, 1, 0]]
        )

        # When
        prob = sp.problem
        solution = sp.solution

        # Then
        self.assertEqual(prob, p)
        self.assertEqual(solution, [[], [0, 1, 1, 0], [1, 0, 1, 0]])

if __name__ == '__main__':
    unittest.main()
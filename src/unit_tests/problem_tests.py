import unittest
import neural_logic_machines.problem as prob

class TestInterpreter(unittest.TestCase):

    def test_problem(self):
        # Given
        problem = prob.Problem(
            max_predicates = [0, 3, 2],
            predicate_names = [[], ['a', 'b', 'c'], ['x', 'y']],
            knowledge_base = ['a(X0) :- b(X0).'],
            objects = ['cat', 'dog']
        )

        # When
        max_predicates = problem.max_predicates
        predicate_names = problem.predicate_names
        knowledge_base = problem.knowledge_base
        objects = problem.objects

        # Then
        self.assertEqual(max_predicates, [0, 3, 2])
        self.assertEqual(predicate_names, [[], ['a', 'b', 'c'], ['x', 'y']])
        self.assertEqual(knowledge_base, ['a(X0) :- b(X0).'])
        self.assertEqual(objects, ['cat', 'dog'])

    def test_solved_problem(self):
        # Given
        problem = prob.Problem(
            max_predicates = [0, 3, 2],
            predicate_names = [[], ['a', 'b', 'c'], ['x', 'y']],
            knowledge_base = ['a(X0) :- b(X0).'],
            objects = ['cat', 'dog']
        )

        sp = prob.SolvedProblem(
            problem = problem,
            solution = [[], [0, 1, 1, 0], [1, 0, 1, 0]]
        )

        # When
        retrieved_problem = sp.problem
        solution = sp.solution

        # Then
        self.assertEqual(retrieved_problem, problem)
        self.assertEqual(solution, [[], [0, 1, 1, 0], [1, 0, 1, 0]])

if __name__ == '__main__':
    unittest.main()
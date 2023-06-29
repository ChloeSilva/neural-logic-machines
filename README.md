# Extending Neural Logic Machines
Implementation of an extension to Neural Logic Machines [(NLM)](https://arxiv.org/pdf/1904.11694.pdf), adding interpretability and recursion.
___
Example use:
```python
from neural_logic_machines import problem

problem = problem.Problem(
    max_predicates = [0, 2, 3],
    max_body = 3,
    predicate_names = [[],
                       ['male', 'female'],
                       ['sibling', 'brother', 'sister']],
    knowledge_base = ['male(X0) :- brother(X0, X1).'],
    objects = ['alice', 'bob', 'carol', 'dave']
)

problem.train('data/training.txt', learning_rate = 0.1)

output = problem.run('data/input.txt', threshold=0.9)
print(output)

rules = problem.rules(threshold=0.9)
print(rules)
```
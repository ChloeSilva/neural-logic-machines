from typing import NamedTuple, List
    
class Problem(NamedTuple):
    max_predicates: List[int]
    predicate_names: List[List[str]]
    knowledge_base: List[str]
    objects: List[str]

class SolvedProblem(NamedTuple):
    problem: Problem
    solution: List[List[int]]
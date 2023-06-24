from itertools import permutations, chain, combinations, cycle
import numpy as np

class Architecture():

    def generate_permutations(self, n):
        return list(permutations(range(n)))
    
    def generate_combinations(self, n):
        l = list(range(n))
        return list(chain.from_iterable(combinations(l, r) for r in range(len(l)+1)))

    def permute_predicate(self, preds):
        perm = self.generate_permutations(preds.ndim - 1)
        return sum([[np.transpose(pred, p) for p in perm] for pred in preds], [])

    def select_body(self, input):
        comb = self.generate_combinations(len(input))
        bodies = [np.array([input[j] for j in i]) for i in comb]
        bodies[0] = np.array([np.full(np.shape(input[0]), 1)])
        return bodies
    
    def apply(self, input):
        output = []
        for (predicates, weights) in input:
            perm = self.permute_predicate(predicates)
            bodies = self.select_body(perm)
            bodies_conj = np.stack([np.prod(body, axis=0) for body in bodies], axis=0)  
            weighted = np.stack([w*b for (w,b) in zip(weights, cycle(bodies_conj))])
            summed = np.minimum(np.sum(np.stack(np.array_split(weighted, len(predicates))), 1), 1)
            output.append(np.maximum.reduce([summed, predicates]))
        
        return np.array(output)
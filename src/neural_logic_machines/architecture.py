from itertools import permutations, chain, combinations
import numpy as np

class Architecture():

    def permute(self, input):
        perm = permutations(range(input.ndim))
        return [np.transpose(input, p) for p in perm]

    def select_body(self, input):
        l = list(range(len(input)))
        comb = chain.from_iterable(combinations(l, r) for r in range(len(l)+1))
        return [np.array([input[j] for j in i]) for i in comb]
from itertools import permutations
import numpy as np

class Architecture():

    def permute(self, input):
        perm = permutations(range(input.ndim))
        return [np.transpose(input, p) for p in perm]
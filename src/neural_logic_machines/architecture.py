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

    def select_body(self, preds):
        comb = self.generate_combinations(len(preds))
        bodies = [np.array([preds[j] for j in i]) for i in comb]
        bodies[0] = np.array([np.full(np.shape(preds[0]), 1)])
        
        return bodies
    
    def expand(self, preds, objects):
        tile_shape = (1, ) * preds.ndim + (objects, )
        final_shape = preds.shape + (objects, )
        return np.reshape(np.tile(preds, tile_shape), final_shape)

    def reduce(self, preds):
        # also consider along axis -1
        return np.concatenate([preds.max(1), preds.min(1)])
    
    def apply(self, weights, premise):
        output = []
        num_premises = [len(p) for p in premise]
        num_objects = premise[-1].shape[-1]
        # expand and reduce predicates of neighbouring arities
        nullary = [np.concatenate((premise[0], self.reduce(premise[1])))]
        mid_arity = [np.concatenate((self.expand(premise[i-1], num_objects), 
                                     premise[i], self.reduce(premise[i+1])))
                     for i in range(1, len(premise) - 1)]
        max_arity = [np.concatenate((self.expand(premise[-2], num_objects), premise[-1]))]
        predicates = nullary + mid_arity + max_arity

        # each iteration is for an different arity of predicates
        for i in range(len(weights)):
            # calculate possible permutations of arguments
            perm = self.permute_predicate(predicates[i])
            # choose possible clause bodies
            bodies = self.select_body(perm)
            # conjunct possible clause bodies
            bodies_conj = np.stack([np.prod(body, axis=0) for body in bodies], axis=0)
            # multiply bodies by weights representing their probabilities
            weighted = np.stack([w*b for (w,b) in zip(weights[i], cycle(bodies_conj))])
            # sums each clause with the same head (capped at 1)
            # TODO: here is where we can add an interesting actiavtion function
            summed = np.minimum(np.sum(np.stack(np.array_split(weighted, len(predicates[i]))), 1), 1)
            # remove expanded predicates from output
            if i > 0:
                summed = summed[num_premises[i-1]:]
                predicates[i] = predicates[i][num_premises[i-1]:]
            # remove reduced predicates from output
            if i < len(weights) - 1:
                summed = summed[:-num_premises[i+1]*2]
                predicates[i] = predicates[i][:-num_premises[i+1]*2]

            # add i-ary predicates to output
            output.append(np.maximum.reduce([summed, predicates[i]]))
        
        return output
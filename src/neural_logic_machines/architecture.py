from itertools import permutations, chain, combinations, cycle
import numpy as np

class Architecture():

    def generate_permutations(self, n):
        return list(permutations(range(n)))
    
    def generate_combinations(self, n, max):
        l = list(range(n))
        return list(chain.from_iterable(
            combinations(l, r) for r in range(min(max+1, n+1))))

    def permute_predicate(self, preds):
        perm = self.generate_permutations(preds.ndim - 1)
        return sum([[np.transpose(pred, p) for p in perm] for pred in preds], [])

    def select_body(self, preds, max):
        comb = self.generate_combinations(len(preds), max)
        bodies = [np.array([preds[j] for j in i]) for i in comb]
        bodies[0] = np.array([np.full(np.shape(preds[0]), 1)])  
        return bodies
    
    def expand(self, preds, objects):
        tile_shape = (1, ) * preds.ndim + (objects, )
        final_shape = preds.shape + (objects, )
        return np.reshape(np.tile(preds, tile_shape), final_shape)

    def reduce(self, preds):
        return preds.max(1)
    
    def hidden_tensor_shape(self, max_pred, max_body):
        i_pred = list(map(sum,zip(
            *[max_pred+[0,0],[0]+max_pred+[0],[0,0]+max_pred])))[1:-1]
        return [max_pred[i] * len(self.generate_combinations(
            len(self.generate_permutations(i))*i_pred[i], max_body)) 
            for i in range(len(max_pred))]
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def d_sigmoid(self, x):
        return x * (1-x)
    
    def apply(self, weights, premise, max_body):
        output = []
        num_premises = [len(p) for p in premise]
        num_objects = premise[-1].shape[-1]
        
        nullary = [np.concatenate((premise[0], self.reduce(premise[1])))]
        mid_arity = [np.concatenate((self.expand(premise[i-1], num_objects), 
                                     premise[i], self.reduce(premise[i+1])))
                     for i in range(1, len(premise) - 1)]
        max_arity = [np.concatenate((self.expand(premise[-2], num_objects), premise[-1]))]
        predicates = nullary + mid_arity + max_arity

        for i in range(len(weights)):
            if len(weights[i]) == 0:
                output.append((np.array([]), np.array([])))
                continue
            # calculate possible permutations of arguments
            perm = self.permute_predicate(predicates[i])
            # choose possible clause bodies
            bodies = self.select_body(perm, max_body)
            # conjunct possible clause bodies
            bodies_conj = np.stack([np.prod(body, axis=0) for body in bodies], axis=0)
            # multiply bodies by weights representing their probabilities
            wb_pair = list(zip(weights[i], cycle(bodies_conj)))
            weighted = np.stack([w*b for (w,b) in wb_pair])
            # sums each clause with the same head
            summed = np.sum(np.stack(np.array_split(weighted, num_premises[i])), 1)
            # apply activation function
            sig = self.sigmoid(summed)
            # sum with facts and add to output
            out = np.maximum.reduce([sig, premise[i]])
            output.append((out, wb_pair))
        
        return output
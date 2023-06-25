import numpy as np
import neural_logic_machines.architecture as architecture

class Interpreter():

    arch = architecture.Architecture()

    def weights_to_predicates(self, weights, problem, threshold):
        clauses = []
        for i in range(len(problem.max_predicates)):
            perm = self.arch.generate_permutations(i)
            comb = self.arch.generate_combinations(
                len(perm)*problem.max_predicates[i])
            
            for j in range(len(weights[i])):
                if weights[i][j] < threshold:
                    continue

                head_pred = j // len(comb)
                body_indices = comb[j % len(comb)]
                body_pred = [b // len(perm) for b in body_indices]
                body_perm = [perm[b] for b in body_indices]
                
                clauses.append(self.clause_index_to_text(
                    i,
                    head_pred,
                    body_pred,
                    body_perm,
                    problem.predicate_names[i]))
                
        return clauses      

    def clause_index_to_text(self, arity, head_pred, body_pred,
                      body_perm, pred_names):
        head_text = pred_names[head_pred] + self.perm_to_text(range(arity))
        clause_text = [pred_names[pred] + self.perm_to_text(perm) 
                       for (pred, perm) in zip(body_pred, body_perm)]
        
        return head_text + ' :- ' + ', '.join(clause_text) + '.'

    def perm_to_text(self, perm):
        return '(' + ', '.join(['X'+str(i) for i in perm]) + ')'

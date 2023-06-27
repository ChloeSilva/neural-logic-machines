import numpy as np
from itertools import takewhile, dropwhile
import neural_logic_machines.architecture as architecture

class Interpreter():

    arch = architecture.Architecture()

    def input_to_tensor(self, input, problem):
        max = problem.max_predicates
        objects = problem.objects
        tensor = [np.zeros((max[i],)+((len(objects),)*i)) 
                  for i in range(len(max))]
        for fact in input:
            head = self.get_head(fact, problem.predicate_names)
            arguments = self.get_arguments(fact, objects)
            index = tuple((head[1],)+tuple(reversed(arguments)))
            tensor[head[0]][index] = 1

        return tensor
    
    def tensor_to_output(self, tensor, problem, threshold):
        output = []
        for layer in range(len(tensor)):
            for pred in range(len(tensor[layer])):
                facts = np.argwhere(tensor[layer][pred] >= threshold)
                for fact in facts:
                    name = problem.predicate_names[layer][pred]
                    args = [problem.objects[arg] for arg in reversed(fact)]
                    output.append(name+'('+', '.join(args)+').')
        
        return output


    def predicates_to_weights(self, problem):
        names = problem.predicate_names
        kb = problem.knowledge_base
        max = problem.max_predicates
        internal_names = [len(n) for n in self.internal(names)]
        output = [[0]*max[i]*len(
            self.arch.generate_combinations(len(
            self.arch.generate_permutations(i))*internal_names[i])) 
            for i in range(len(max))]
        
        for rule in kb:
            head = self.get_head(rule, names)
            body = self.get_body(rule, names, head[0])
            perm = self.arch.generate_permutations(head[0])
            comb = self.arch.generate_combinations(
                len(perm)*internal_names[head[0]])
            comb_i = tuple([(self.p_index(n,head[0],max))*len(perm)+perm.index(p) 
                            for (n, p) in body])
            output[head[0]][head[1]*len(comb)+comb.index(comb_i)] = 1
        
        return np.array(output, dtype=object)

    def weights_to_predicates(self, solved_problem, threshold):
        problem = solved_problem.problem
        weights = solved_problem.solution
        internal_names = self.internal(problem.predicate_names)
        max = problem.max_predicates

        clauses = []
        for i in range(len(max)):
            perm = self.arch.generate_permutations(i)
            comb = self.arch.generate_combinations(
                len(perm)*len(internal_names[i]))
            
            tags = ['normal']*max[i]
            if i > 0: tags = ['expanded']*max[i-1] + tags
            if i < len(max)-1: tags += ['reduced']*max[i+1]
            
            for j in range(len(weights[i])):
                if weights[i][j] < threshold:
                    continue
                
                head_pred = j // len(comb)
                if i != 0: head_pred += max[i-1]
                body_indices = comb[j % len(comb)]
                body_pred = [b//len(perm) for b in body_indices]
                body_pred = [(b, tags[b]) for b in body_pred]
                body_perm = [perm[b % len(perm)] for b in body_indices]
                
                clauses.append(self.clause_index_to_text(
                    i,
                    head_pred,
                    body_pred,
                    body_perm,
                    internal_names[i]))
                
        return clauses
    
    def internal(self, names):
        new = [[], []] + names, [[]] + names + [[]], names + [[], []]
        return [sum(n, []) for n in zip(*new)][1:-1]

    def clause_index_to_text(self, arity, head_pred, body_pred,
                      body_perm, pred_names):
        head_text = pred_names[head_pred] + self.perm_to_text(range(arity), "normal")
        clause_text = [pred_names[pred] + self.perm_to_text(perm, tag) 
                       for ((pred,tag), perm) in zip(body_pred, body_perm)]
        
        return head_text + ' :- ' + ', '.join(clause_text) + '.'

    def perm_to_text(self, perm, tag):
        if tag == 'reduced':
            perm += (len(perm), )
            text = '(' + ', '.join(['X'+str(i) for i in perm]) + ')'
        elif tag == 'expanded':
            text = '(' + ', '.join(['X'+str(i) for i in perm[:-1]]) + ')'
        else:
            text = '(' + ', '.join(['X'+str(i) for i in perm]) + ')'
        
        return text
    
    def name_to_index(self, rule, names):
        for i in range(len(names)):
            for j in range(len(names[i])):
                if names[i][j] == rule:
                    return (i, j)
                
    def p_index(self, index, layer, max):
        if index[0] == layer-1:
            return index[1]
        if index[0] == layer:
            return max[layer-1] + index[1]
        return max[layer-1] + max[layer] + index[1]
                
    def get_arguments(self, fact, objects):
        args = list(dropwhile(lambda x: x != '(', fact))[1:-2]
        return [objects.index(a) for a in ''.join(args).split(', ')]


    def get_head(self, rule, names):
        string = list(takewhile(lambda x: x != '(', rule))
        return self.name_to_index(''.join(string), names)
        
                
    def get_body(self, rule, names, layer):
        rule = list(dropwhile(lambda x: x != '-', rule))[2:]
        body = []
        name = ''
        i = 0
        while i < len(rule):
            # get the index of the predicate
            while rule[i] != '(':
                name += rule[i]
                i += 1
            # get the indices of the arguments
            arguments = ()
            while rule[i] != ')':
                i += 2
                argument = ''
                while rule[i].isdigit():
                    argument += rule[i]
                    i += 1
                arguments += (int(argument), )
                if rule[i] == ',':
                    i += 1
            i += 3
            if len(arguments) < layer:
                arguments += (layer-1, )
            body.append((self.name_to_index(name, names), arguments[:layer]))
            name = ''

        return body

        
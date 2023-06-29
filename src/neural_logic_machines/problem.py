import neural_logic_machines.architecture as architecture
import neural_logic_machines.interpreter as interpreter
from itertools import groupby
import random
import numpy as np

class Problem():

    solution = []

    def __init__(self, max_predicates, max_body,
                 predicate_names, knowledge_base, objects):
        self.max_predicates = max_predicates
        self.max_body = max_body
        self.predicate_names = predicate_names
        self.knowledge_base = knowledge_base
        self.objects = objects
        self.arch = architecture.Architecture()
        self.interpret = interpreter.Interpreter()

    def train(self, training_path, learning_rate, iterations=1):
        with open(training_path) as f:
            training_data = [line.strip() for line in f]

        processed = self.process(training_data)
        in_out = [(self.interpret.input_to_tensor(i, self),
                     self.interpret.input_to_tensor(o, self))
                    for i, o in processed]

        weights = [[random.random() for _ in range(i)] 
                   for i in self.arch.hidden_tensor_shape(
                   self.max_predicates, self.max_body)]

        for _ in range(iterations):
            for input, goal in in_out:
                result = self.arch.apply(weights, input, self.max_body)
                
                for i in range(len(weights)):
                    preds = self.max_predicates[i]
                    if preds == 0: continue
                    out, wb_pair = result[i]
                    d_a = [np.maximum(b, 0.01) for (w,b) in wb_pair]
                    weighted = np.stack([w*b for (w,b) in wb_pair])
                    d_sig = self.arch.d_sigmoid(np.sum(np.stack(np.array_split(weighted, preds)), 1))
                    d_cost = out - goal[i]
                    delta = np.tile(d_cost,(preds,)+(1,)*i) * np.tile(d_sig,(preds,)+(1,)*i) * d_a
                    avg_delta = np.average(delta.reshape((delta.shape[0], -1)), 1)
                    weights[i] += avg_delta * -learning_rate

        self.solution = weights
        return weights
            
    def process(self, d):
        d = list(filter(lambda z: z != '', d))
        d = [tuple(y) for x, y in groupby(d, lambda z: z == 'in:') if not x]
        d = [[list(y) for x, y in groupby(p, lambda z: z == 'out:') if not x] for p in d]
        return d
    
    def run(self, input, threshold):
        with open(input) as f:
            input = [line.strip() for line in f]

        input = self.process(input)
        tensor = self.interpret.input_to_tensor(input[0][0], self)
        result = self.arch.apply(self.solution, tensor, self.max_body)
        return self.interpret.tensor_to_output(list(zip(*result))[0], self, threshold)
    
    def rules(self, threshold):
        return self.interpret.weights_to_predicates(self, threshold)
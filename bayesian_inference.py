from pgmpy.inference import BeliefPropagation

class BayesianInference:
    def __init__(self, model):
        self.model = model
        self.infer = BeliefPropagation(model)
        self.infer.calibrate()

    def map_query(self, variables, evidence):
        return self.infer.map_query(variables=variables, evidence=evidence)

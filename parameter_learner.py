from pgmpy.estimators import BayesianEstimator

class ParameterLearner:
    def __init__(self, model, df):
        self.model = model
        self.df = df

    def fit(self, prior_type='BDeu'):
        estimator = BayesianEstimator(self.model, self.df)
        self.model.fit(self.df, estimator=BayesianEstimator, prior_type=prior_type)
        return self.model

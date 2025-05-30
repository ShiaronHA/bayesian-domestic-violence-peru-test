from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore, BDeuScore, MmhcEstimator, PC

class StructureLearner:
    def __init__(self, df):
        self.df = df
        self.model = None

    def learn(self, algorithm='hill_climb', scoring_method=None, columns_to_exclude=None):
        if algorithm == 'hill_climb':
            est = HillClimbSearch(self.df)
            if scoring_method == 'bic':
                model = est.estimate(scoring_method=BicScore(self.df), max_iter=5000, max_indegree=5)
            elif scoring_method == 'bdeu':
                model = est.estimate(scoring_method=BDeuScore(self.df), max_indegree=5, max_iter=int(1e4))
            elif scoring_method == 'k2':
                model = est.estimate(scoring_method='k2score', max_indegree=5, max_iter=int(1e4))
            else:
                raise ValueError("Scoring method no soportado para Hill Climbing.")
            bn_model = BayesianNetwork(model.edges())
        elif algorithm == 'pc':
            est = PC(self.df)
            model = est.estimate(ci_test='chi_square', variant="stable", max_cond_vars=4, return_type='dag')
            bn_model = BayesianNetwork(model.edges())
        elif algorithm == 'mmhc':
            if columns_to_exclude:
                mmhc_df = self.df.drop(columns=[col for col in columns_to_exclude if col in self.df.columns])
            else:
                mmhc_df = self.df
            mmhc = MmhcEstimator(mmhc_df)
            skeleton = mmhc.mmpc()
            hc = HillClimbSearch(mmhc_df)
            model = hc.estimate(
                tabu_length=5,
                white_list=skeleton.to_directed().edges(),
                scoring_method=BDeuScore(mmhc_df),
                max_indegree=3,
                max_iter=100
            )
            bn_model = BayesianNetwork(model.edges())
        else:
            raise ValueError("Algoritmo no soportado.")
        self.model = bn_model
        return bn_model

from pgmpy.estimators import BayesianEstimator


def parameter_learning(model, df):
    """
    Estimate parameters for a Bayesian Network using BayesianEstimator.
    """
    print("\nEstimating parameters with BayesianEstimator...")
    estimator = BayesianEstimator(model, df)
    model.fit(df, BayesianEstimator, prior_type='BDeu')
    return model

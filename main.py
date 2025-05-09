import pandas as pd
import pickle
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore, BDeuScore, MmhcEstimator


def preprocess_data(filepath):
    df = pd.read_csv(filepath, delimiter=',')
    print("Forma inicial del DataFrame:", df.shape)

    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        df[col] = df[col].astype('category')

    df = df.dropna()
    for col in df.columns:
        df[col] = df[col].astype('category').cat.codes

    return df


def learn_structure(df, algorithm='hill_climb', scoring_method=None, output_path=None):
    if algorithm == 'hill_climb':
        print(f"\nAprendiendo con Hill Climbing usando {scoring_method}...")
        est = HillClimbSearch(df)

        if scoring_method == 'bic':
            model = est.estimate(scoring_method=BicScore(df))
        elif scoring_method == 'k2':
            model = est.estimate(scoring_method='k2score', max_indegree=4, max_iter=int(1e4))
        else:
            raise ValueError("Scoring method no soportado para Hill Climbing.")

        bn_model = BayesianNetwork(model.edges())

    elif algorithm == 'mmhc':
        print("\nAprendiendo con MMHC (Max-Min Hill Climbing)...")
        mmhc = MmhcEstimator(df)
        skeleton = mmhc.mmpc()
        hc = HillClimbSearch(df)
        model = hc.estimate(
            tabu_length=10,
            white_list=skeleton.to_directed().edges(),
            scoring_method=BDeuScore(df)
        )
        bn_model = BayesianNetwork(model.edges())
    else:
        raise ValueError("Algoritmo no soportado.")

    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(bn_model, f)
        print(f"Modelo guardado en '{output_path}'")

    print("Estructura aprendida:", bn_model.edges())
    return bn_model


def main():
    df = preprocess_data('data/df_processed.csv')

    learn_structure(df, algorithm='hill_climb', scoring_method='bic',
                    output_path='./uploads/model_structure_29_hillClimbing_29.pkl')

    learn_structure(df, algorithm='hill_climb', scoring_method='k2',
                    output_path='./uploads/model_structure_29_hillClimbing_k2.pkl')

    # learn_structure(df, algorithm='mmhc',
    #                 output_path='./uploads/model_structure_29_mmhc.pkl')


if __name__ == "__main__":
    main()

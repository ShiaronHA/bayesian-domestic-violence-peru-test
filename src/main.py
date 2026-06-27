"""
Full pipeline orchestrator for the Bayesian Network domestic violence risk model.
Run this file to reproduce the complete experimental pipeline in order.
"""
from data_preprocessor import build_and_save_processed_dataset, split_and_save_datasets
from structure_learner import main as run_structure_learning
from bayesian_inference import main as run_bayesian_inference
from random_forest import main as run_random_forest
from decision_tree import main as run_decision_tree
from expert_in_the_loop import main as run_expert_in_the_loop


def main():
    """
    Runs the full research pipeline:
    1. Preprocess raw SAV data into train/val datasets
    2. Learn Bayesian Network structure (multiple algorithms and sample sizes)
    3. Run Bayesian Network inference and evaluate
    4. Train and evaluate Random Forest baseline
    5. Train and evaluate Decision Tree baseline
    6. Run Expert-in-the-Loop structure learning with LLM
    """
    # Step 1: Data preprocessing
    processed_csv = build_and_save_processed_dataset()
    if processed_csv:
        split_and_save_datasets(processed_csv)

    # Step 2: Bayesian Network structure learning
    run_structure_learning()

    # Step 3: Bayesian Network inference and evaluation
    run_bayesian_inference()

    # Step 4-5: Baseline classifiers
    run_random_forest()
    run_decision_tree()

    # Step 6: Expert-in-the-Loop structure learning with LLM (requires GEMINI_API_KEY)
    run_expert_in_the_loop()


if __name__ == "__main__":
    main()

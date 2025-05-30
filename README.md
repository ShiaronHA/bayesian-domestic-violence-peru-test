# Bayesian Network Analysis of Domestic Violence in Peru

This project aims to analyze domestic violence data in Peru using Bayesian Networks. The pipeline involves data preprocessing, structure learning, parameter learning, and inference.

## Project Structure

- `main.py`: Main script to run the Bayesian Network learning pipeline.
- `data_preprocessor.py`: Script for preprocessing the raw data.
- `structure_learner.py`: Module for learning the structure of the Bayesian Network.
- `parameter_learner.py`: Module for learning the parameters of the Bayesian Network.
- `bayesian_inference.py`: Module for performing inference on the learned network.
- `data/`: Directory for data files.
    - `input_data/`: Contains raw data files.
    - `df_processed.csv`: Processed data used for model training.
- `models/`: Directory for storing trained model files.
- `dag/`: Directory for storing visualizations of learned network structures.
- `uploads/`: Directory for various output files, including results and mappings.

## Data

The raw data for this project is not stored in this GitHub repository due to its size and format. It can be downloaded from the following Google Drive link:

[Raw Data Files](https://drive.google.com/drive/folders/1Ge8z7mlQg2qGoBehYEhLS8oN5sAhvAQx?usp=drive_link)

The raw data includes `.sav`, `.xlsx`, and `.csv` files. These files are processed by `data_preprocessor.py` to generate `data/df_processed.csv`, which is then used by the main pipeline.

## Setup and Execution

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```
2.  **Download Data:**
    Download the raw data files from the Google Drive link provided above and place them into the `data/input_data/` directory.

3.  **Set up Python Environment:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run the pipeline:**
    ```bash
    python main.py
    ```

## Contributing

Please refer to the project's issue tracker for areas where contributions are welcome.

## License


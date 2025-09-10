# Molecular Property Prediction using Machine Learning

## Project Overview
This project demonstrates how machine learning can accelerate scientific research by predicting molecular properties without running expensive simulations. The goal is to create a fast and efficient alternative to traditional computational methods.

## The Problem
Traditional molecular simulations are computationally expensive and can take hours or days to run for a single molecule. This bottleneck severely limits the number of molecules that can be screened for potential applications in fields like drug discovery.

## The Solution
This project uses a Random Forest Regressor model trained on a large, real-world dataset (ESOL) to predict molecular properties. The model learns the relationship between a molecule's basic features and its solubility, enabling instantaneous predictions.

## Technologies Used
- **Python:** The core programming language for the project.
- **scikit-learn:** Used for building and training the machine learning model.
- **pandas:** For data manipulation and analysis.
- **deepchem & RDKit:** For handling and processing the molecular dataset.

## How to Run the Project
1.  **Install Libraries:** `pip install scikit-learn pandas deepchem rdkit`
2.  **Run scripts in order:**
    - `python prepare_esol_data.py` (Downloads and prepares the dataset)
    - `python train_model.py` (Trains the ML model)
    - `python predict_solubility.py` (Uses the trained model to make a prediction)

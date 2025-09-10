import joblib
import pandas as pd

try:
    model = joblib.load('solubility_predictor_model.pkl')
except FileNotFoundError:
    print("Error: The trained model file was not found. Please run 'train_model.py' first.")
    exit()

new_molecule_features = {
    'molecular_weight': [150.2],
    'logP': [2.5],
    'tpsa': [40.0]
}

new_molecule_df = pd.DataFrame(new_molecule_features)

predicted_solubility = model.predict(new_molecule_df)

print(f"Predicting solubility for a new molecule:")
print(new_molecule_df)
print(f"\nPredicted Solubility: {predicted_solubility[0]:.4f}")
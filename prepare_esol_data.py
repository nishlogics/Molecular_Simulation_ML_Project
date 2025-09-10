import deepchem as dc
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

def get_molecular_descriptors(smiles):
    """Calculates molecular descriptors from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'molecular_weight': Descriptors.MolWt(mol),
        'logP': Descriptors.MolLogP(mol),
        'tpsa': Descriptors.TPSA(mol)
    }

tasks, datasets, transformers = dc.molnet.load_esol()
train_dataset, val_dataset, test_dataset = datasets

df_train = train_dataset.to_dataframe()
df_val = val_dataset.to_dataframe()
df_test = test_dataset.to_dataframe()

df = pd.concat([df_train, df_val, df_test])

df = df.rename(columns={'y': 'solubility'})
df = df.reset_index(drop=True)

descriptors_list = []
for smiles in df['X']:
    descriptors = get_molecular_descriptors(smiles)
    descriptors_list.append(descriptors if descriptors else {})

descriptors_df = pd.DataFrame(descriptors_list)
df = pd.concat([df, descriptors_df], axis=1)

df = df.dropna(subset=['molecular_weight'])

df = df[['molecular_weight', 'logP', 'tpsa', 'solubility']]
df.to_csv('esol_molecular_data.csv', index=False)

print("Downloaded and prepared 'esol_molecular_data.csv'.")
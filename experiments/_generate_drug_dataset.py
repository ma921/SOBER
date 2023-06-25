import os
import torch
import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles, AllChem, Descriptors

def create_malaria_dataset(data_path):
    """
    Create malaria dataset
    
    Args:
    - data_path: string, the data path to the file "malaria_box_dataset.csv"
    
    Returns:
    - data: list, the list of 2,048 binary features and target values (Activity (EC50 uM)).
    """
    df = pd.read_csv(data_path)
    bond_radius = 3
    nBits = 2048
    smiles = np.asarray(df["Canonical_Smiles"])
    true_targets = torch.from_numpy(np.asarray(df["Activity (EC50 uM)"]))
    true_targets = -1 * true_targets  # convert minimization task to maximization
    rdkit_mols = [MolFromSmiles(smile) for smile in smiles]
    fps = [
        AllChem.GetMorganFingerprintAsBitVect(mol, bond_radius, nBits=nBits)
        for mol in rdkit_mols
    ]
    features = torch.from_numpy(np.asarray(fps))
    
    return (
        features.float(),
        true_targets.float(),
    )

def create_solvent_dataset(data_path):
    """
    Create solvent dataset
    
    Args:
    - data_path: string, the data path to the file "QM9_dipole.csv"
    
    Returns:
    - data: list, the list of 2,048 binary features and target values (dipole [debye]).
    """
    df = pd.read_csv(data_path)
    bond_radius = 3
    nBits = 2048
    smiles = np.asarray(df["smiles"])
    true_targets = torch.from_numpy(np.asarray(df["dipole"]))
    rdkit_mols = [MolFromSmiles(smile) for smile in smiles]
    fps = [
        AllChem.GetMorganFingerprintAsBitVect(mol, bond_radius, nBits=nBits)
        for mol in rdkit_mols
    ]
    features = torch.from_numpy(np.asarray(fps))
    
    return (
        features.float(),
        true_targets.float(),
    )

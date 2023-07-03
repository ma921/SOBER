import os
import torch
import numpy as np
import itertools
from abc import abstractmethod
from SOBER._prior import DatasetPrior
from experiments._generate_drug_dataset import create_solvent_dataset


def featurise_dataset(data_path):
    #SOLVENT_DIR_NAME = "./experiments/dataset/"
    # dataset can be downloaded in http://quantum-machine.org/datasets/ 
    #data_filename = "QM9_dipole.csv"
    #data_path = os.path.join(SOLVENT_DIR_NAME, data_filename)
    features, true_targets = create_solvent_dataset(data_path)
    return features, true_targets

def setup_solvent(data_path):
    """
    Set up the experiments with solvent materials discovery task
    
    Return:
    - prior: class, the function of dataset prior
    """
    features, true_targets = featurise_dataset(data_path)
    prior = DatasetPrior(features, true_targets)
    
    return prior

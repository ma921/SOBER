import os
import torch
import numpy as np
import itertools
from abc import abstractmethod
from SOBER._prior import DatasetPrior
from experiments._generate_drug_dataset import create_malaria_dataset


def featurise_dataset(data_path):
    #MALARIA_DIR_NAME = "../experiments/dataset/"
    # dataset can be downloaded in  https://www.mmv.org/mmv-open/malaria-box/malaria-box-supporting-information
    #data_filename = "malaria_box_dataset.csv"
    #data_path = os.path.join(MALARIA_DIR_NAME, data_filename)
    features, true_targets = create_malaria_dataset(data_path)
    return features, true_targets
    
def setup_malaria(data_path):
    """
    Set up the experiments with anti-malarial drug discovery task
    
    Return:
    - prior: class, the function of binary prior
    """
    features, true_targets = featurise_dataset(data_path)
    prior = DatasetPrior(features, true_targets)
    
    return prior

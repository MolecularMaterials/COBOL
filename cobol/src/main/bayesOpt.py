import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch import get_best_candidates
import sys
utils_path = ('../utils')
sys.path.append(utils_path)
from chem_utils import get_fingerprint

class BayesianOptimizer:

    def __init__(self, initial_data, known_results=None, device=None):
        self.device = device or torch.device("cpu")
        self.initial_data = initial_data
        self.known_results = known_results
        self.train_X, self.train_Y = self.preprocess_data(initial_data)

    def preprocess_data(self, data):
        smilesList, target_values = zip(*data)
        features = []
        #features = Smiles2Fingerprint(smilesList, 'PhysChemFeats')
        for s in smilesList:
            _, feats = get_fingerprint(s, 'PhysChemFeats')
            features.append(feats)
        return torch.tensor(features, device=self.device), torch.tensor(target_values, device=self.device).unsqueeze(-1)

    def initialize_model(self):
        self.model = SingleTaskGP(self.train_X, self.train_Y)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)

    def acquire_candidate(self):
        EI = ExpectedImprovement(self.model, self.train_Y.max())
        bounds = torch.stack([self.train_X.min(dim=0).values, self.train_X.max(dim=0).values])
        candidate, acq_value = optimize_acqf(EI, bounds=bounds, q=1, num_restarts=5, raw_samples=20)
        return candidate

    def dummy_experiment(self, candidate):
        _, smiles = get_fingerprint(candidate, 'PhysChemFeats')
        # Lookup the pre-computed result for this smiles string
        return self.known_results[smiles]

    def optimize(self, iterations):
        self.initialize_model()

        for i in range(iterations):
            candidate = self.acquire_candidate()
            new_y = dummy_experiment(candidate)
            self.train_X = torch.cat([self.train_X, candidate])
            self.train_Y = torch.cat([self.train_Y, new_y])

            self.initialize_model()

        best_candidate = get_best_candidates(self.train_X, self.train_Y, num_candidates=1)
        return best_candidate

if __name__ == '__main__':
    import pandas as pd
    url = 'https://github.com/MolecularMaterials/COBOL/blob/main/cobol/CaseStudy/single-objective-HBE/Oxidation-HBE-b3lyp-results-clean.csv?raw=true'

    # Load the data into a pandas DataFrame
    data = pd.read_csv(url)

    # Extract SMILES strings and target values
    smiles_strings = data['canonSMILES'].values
    target_values = data['Potential'].values

    # Combine the SMILES strings and target values into tuples
    initial_data = list(zip(smiles_strings, target_values))

    # Create an instance of BayesianOptimizer
    optimizer = BayesianOptimizer(initial_data)

    # Perform optimization
    best_candidate = optimizer.optimize(iterations=20)

    # Convert the best candidate back to SMILES (you need a function for this)
    best_smiles = features_to_smiles(best_candidate)

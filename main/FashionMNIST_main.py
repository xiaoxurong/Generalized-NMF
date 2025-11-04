import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import re
import numpy as np
import argparse
import wandb
from scipy.io import loadmat
from src.GenNMF import *
from src.modified_dscnmf import *
from src.baseline import *
from src.deepNMF import *
from src.deepSSCNMF import *
from sklearn.preprocessing import normalize
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def arg_parser():
    parser = argparse.ArgumentParser(description="Iterative subspace clustering with NMF")
    # parser.add_argument('--m', type=int, default=50, help='Dimension of the ambient space (default: 50)')
    parser.add_argument('--r', type=int, default=5, help='Dimension (rank) of each subspace (default: 5)')
    parser.add_argument('--n', type=int, default=50, help='Number of points per subspace (default: 100)')
    parser.add_argument('--K', type=int, default=10, help='Number of subspaces (default: 10)')
    parser.add_argument('--sigma', type=float, default=0.0, help='Standard deviation of Gaussian noise (default: 0.0)')
    parser.add_argument('--alpha', type=float, default=1e-2, help='Regularization parameter for ssc')
    parser.add_argument('--max_iter', type=int, default=200, help='Maximum number of iterations (default: 50)')
    parser.add_argument('--tol', type=float, default=1e-6, help='Tolerance for stopping criterion (default: 1e-6)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for clustering (default: None)')
    parser.add_argument('--model', type=str, choices=['sscnmf', 'ricc', 'gnmf', 'gpcanmf', 'onmf_relu', 'dscnmf', 'onmf', 'deepnmf', 'deepsscnmf', 'ssc-omp-nmf', 'oraclenmf'],
                        help='Model to use for clustering')
    parser.add_argument('--n_nonzero_coefs', type=int, default=8, help='Number of non-zero coefficients for OMP (default: 8)')
    parser.add_argument('--l1_reg', type=float, default=0.01,
                        help='L1 regularization parameter for ONMF-ReLU/GPCANMF')
    return parser.parse_args()

def main(model, r, n, K, sigma=0.0, alpha=0.1, l1_reg=0.01, random_state=None, max_iter=50, tol=1e-6, n_nonzero_coefs=8):
    transform = transforms.ToTensor()
    full_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    X_full = full_data.data.numpy().reshape(len(full_data), -1) / 255.0
    y_full = full_data.targets.numpy()

    # Subsample 1000 examples
    subset_idx = np.random.choice(len(X_full), size=1000, replace=False)
    X = X_full[subset_idx]
    true_labels = y_full[subset_idx]

    if sigma > 0:
        # Add non-negative Gaussian noise to the data
        noise = np.random.normal(0, sigma, X.shape)
        X += noise
        X = normalize(X, axis=1)
    X = X.T  # shape (features, samples)

    if model == 'sscnmf':
        project_name = 'sscnmf-FashionMNIST'
        acc, ARI, NMI, reconstruction_error, _, _, _ = ssc_nmf_baseline(
            X, K=K, r=r, true_labels=true_labels, alpha=alpha, random_state=random_state)
    elif model == 'ssc-omp-nmf':
        project_name = 'ssc-omp-nmf-FashionMNIST'
        acc, ARI, NMI, reconstruction_error, _, _, _ = ssc_omp_nmf_baseline(
            X, K=K, r=r, true_labels=true_labels, n_nonzero_coefs=n_nonzero_coefs, random_state=random_state)
    elif model == 'ricc':
        project_name = 'ricc-FashionMNIST'
        acc, ARI, NMI, reconstruction_error, _, _, _ = iter_reg_coneclus_warmstart(
            X, K=K, r=r, true_labels=true_labels,
            alpha=alpha, max_iter=max_iter, NMF_method='anls', ord=2, random_state=random_state)
    elif model == 'gnmf':
        project_name = 'gnmf-FashionMNIST'
        acc, ARI, NMI, reconstruction_error, _, _, _ = GNMF_clus(
            X, K=K, r=r, true_labels=true_labels, max_iter=max_iter)
    elif model == 'gpcanmf':
        project_name = 'gpcanmf-FashionMNIST'
        acc, ARI, NMI, reconstruction_error, _, _, _ = gpca_nmf(
            X, K=K, r=r, true_labels=true_labels, random_state=random_state)
    elif model == 'dscnmf':
        project_name = 'dscnmf-FashionMNIST'
        acc, ARI, NMI, reconstruction_error = dsc_nmf_baseline(
            X, K=K, r=r, true_labels=true_labels)
    elif model == 'onmf':
        project_name = 'onmf-FashionMNIST'
        acc, ARI, NMI, reconstruction_error = onmf_ding(
            X, K=K, true_labels=true_labels, random_state=random_state)
    elif model == 'deepnmf':
        project_name = 'deepnmf-FashionMNIST'
        acc, ARI, NMI, reconstruction_error = dsc_nmf_baseline(
            X, K=K, r=r, true_labels=true_labels)
    elif model == 'deepsscnmf':
        project_name = 'deepsscnmf-FashionMNIST'
        acc, ARI, NMI, reconstruction_error = deep_ssc_nmf(
            X, ranks=[256, 128, 64], alpha=alpha, n_iter=max_iter,
            true_labels=true_labels)
    elif model == 'oraclenmf':
        project_name = 'oraclenmf-FashionMNIST'
        acc = 1.0
        ARI = 1.0
        NMI = 1.0
        r_oracle = r // K
        _, _, reconstruction_error = oracle_nmf(
            X, K=K, r=r_oracle, true_labels=true_labels)
    else:
        raise ValueError(f"Unknown model: {model}") 
    
    print(f"Received model: {model}")
    wandb.init(
        project="coneClustering",
        name=project_name
    )
    wandb.log({
        "accuracy": acc,
        "ARI": ARI,
        "NMI": NMI,
        "reconstruction_error": reconstruction_error
    })
    print("\n--- Results ---")
    print(f"Accuracy: {acc}")
    print(f"ARI: {ARI}")
    print(f"NMI: {NMI}")
    print(f"Reconstruction Error: {reconstruction_error}")  

    wandb.finish()
if __name__ == "__main__":
    args = arg_parser()
    main(model=args.model, r=args.r, n=args.n, K=args.K, sigma=args.sigma,
         alpha=args.alpha, l1_reg=args.l1_reg, random_state=args.random_state,
         max_iter=args.max_iter, tol=args.tol, n_nonzero_coefs=args.n_nonzero_coefs)
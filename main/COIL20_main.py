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
from sklearn.decomposition import PCA

def arg_parser():
    parser = argparse.ArgumentParser(description="Iterative subspace clustering with NMF")
    # parser.add_argument('--m', type=int, default=50, help='Dimension of the ambient space (default: 50)')
    parser.add_argument('--r', type=int, default=5, help='Dimension (rank) of each subspace (default: 5)')
    parser.add_argument('--n', type=int, default=50, help='Number of points per subspace (default: 100)')
    parser.add_argument('--K', type=int, default=20, help='Number of subspaces (default: 20)')
    parser.add_argument('--sigma', type=float, default=0.0, help='Standard deviation of Gaussian noise (default: 0.0)')
    parser.add_argument('--alpha', type=float, default=1e-2, help='Regularization parameter for ssc')
    parser.add_argument('--max_iter', type=int, default=200, help='Maximum number of iterations (default: 50)')
    parser.add_argument('--tol', type=float, default=1e-6, help='Tolerance for stopping criterion (default: 1e-6)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for clustering (default: None)')
    parser.add_argument('--model', type=str, choices=['sscnmf', 'ricc', 'gnmf', 'gpcanmf', 'onmf_relu', 'dscnmf', 'onmf', 'deepnmf', 'deepsscnmf', 'ssc-omp-nmf', 'oraclenmf'],
                        help='Model to use for clustering')
    parser.add_argument('--l1_reg', type=float, default=0.01,
                        help='L1 regularization parameter for ONMF-ReLU/GPCANMF')
    parser.add_argument('--n_nonzero_coefs', type=int, default=8, help='Number of non-zero coefficients for OMP')
    return parser.parse_args()

def main(model, r, n, K, sigma=0.0, alpha=0.1, l1_reg=0.01, random_state=None, max_iter=50, tol=1e-6, n_nonzero_coefs=8):
    coil20_data = loadmat('data/COIL20.mat')
    X_full = coil20_data['fea']  # (1440, 1024)
    true_labels = coil20_data['gnd'].flatten() - 1

    np.random.seed(random_state)

    pca = PCA(n_components=300, whiten=True)
    X_pca = pca.fit_transform(X_full)         # (1440, 300)
    X = X_pca.T                               # (300, 1440)
    X = normalize(X, axis=0)
    # After PCA and normalization
    X_min = X.min()
    if X_min < 0:
        X = X - X_min + 1e-8  # shift so the smallest entry is slightly above zero


    if sigma > 0:
        # Add non-negative Gaussian noise to the data
        noise = np.random.normal(0, sigma, X.shape)
        X += noise
        X = normalize(X, axis=0)
    
    if model == 'sscnmf':
        project_name = 'sscnmf-COIL20'
        acc, ARI, NMI, reconstruction_error, _, _, _ = ssc_nmf_baseline(
            X, K=K, r=r, true_labels=true_labels, alpha=alpha)
    elif model == 'ssc-omp-nmf':
        project_name = 'ssc-omp-nmf-COIL20'
        acc, ARI, NMI, reconstruction_error, _, _, _ = ssc_omp_nmf_baseline(
            X, K=K, r=r, true_labels=true_labels, n_nonzero_coefs=n_nonzero_coefs, random_state=random_state)
    elif model == 'ricc':
        project_name = 'ricc-COIL20'
        acc, ARI, NMI, reconstruction_error, _, _, _ = iter_reg_coneclus_warmstart(
            X, K=K, r=r, true_labels=true_labels,
            alpha=alpha, max_iter=max_iter, NMF_method='anls', ord=2, random_state=random_state)
    elif model == 'gnmf':
        project_name = 'gnmf-COIL20'
        acc, ARI, NMI, reconstruction_error, _, _, _ = GNMF_clus(
            X, K=K, r=r, true_labels=true_labels)
    elif model == 'gpcanmf':
        project_name = 'gpcanmf-COIL20'
        acc, ARI, NMI, reconstruction_error, _, _, _ = gpca_nmf(
            X, K=K, r=r, true_labels=true_labels,
            l1_reg=l1_reg, random_state=random_state)
    elif model == 'dscnmf':
        project_name = 'dscnmf-COIL20'
        acc, ARI, NMI, reconstruction_error = dsc_nmf_baseline(
            X, K=K, r=r, true_labels=true_labels)
    elif model == 'onmf':
        project_name = 'onmf-COIL20'
        acc, ARI, NMI, reconstruction_error = onmf_ding(
            X, K=K, true_labels=true_labels, random_state=random_state)
    elif model == 'deepnmf':
        project_name = 'deepnmf-COIL20'
        acc, ARI, NMI, reconstruction_error, _, _ = deep_nmf(
            X, random_state=random_state, true_labels=true_labels)
    elif model == 'deepsscnmf':
        project_name = 'deepsscnmf-COIL20'
        acc, ARI, NMI, reconstruction_error = deep_ssc_nmf(
            X, ranks=[256, 128, 64], alpha=alpha, n_iter=max_iter,
            true_labels=true_labels)
    elif model == 'oraclenmf':
        acc = 1.0
        ARI = 1.0
        NMI = 1.0
        r_oracle = r // K
        project_name = 'oraclenmf-COIL20'
        _, _, reconstruction_error = oracle_nmf(
            X, K=K, r=r_oracle, true_labels=true_labels)
    else:
        raise ValueError(f"Unknown model: {model}")
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
    print(f"Clustering Accuracy: {acc:.4f}")
    print(f"Adjusted Rand Index (ARI): {ARI:.4f}")
    print(f"Normalized Mutual Information (NMI): {NMI:.4f}")
    print(f"Final Reconstruction Loss: {reconstruction_error:.4f}")
    wandb.finish()  

if __name__ == "__main__":
    args = arg_parser()
    model = args.model
    r = args.r
    n = args.n
    K = args.K
    sigma = args.sigma
    alpha = args.alpha
    l1_reg = args.l1_reg
    max_iter = args.max_iter
    tol = args.tol
    random_state = args.random_state
    n_nonzero_coefs = args.n_nonzero_coefs

    main(model, r, n, K, sigma, alpha, l1_reg, random_state, max_iter, tol, n_nonzero_coefs)
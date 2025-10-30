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
import pandas as pd
from src.GenNMF import *
from src.modified_dscnmf import *
from src.baseline import *
from src.deepNMF import *
from src.deepSSCNMF import *
from sklearn.preprocessing import normalize, LabelEncoder

def arg_parser():
    parser = argparse.ArgumentParser(description="Iterative subspace clustering with NMF")
    parser.add_argument('--r', type=int, default=5, help='Dimension (rank) of each subspace (default: 5)')
    parser.add_argument('--n', type=int, default=50, help='Number of points per subspace (default: 100)')
    parser.add_argument('--K', type=int, default=7, help='Number of subspaces (default: 7)')
    parser.add_argument('--sigma', type=float, default=0.0, help='Standard deviation of Gaussian noise (default: 0.0)')
    parser.add_argument('--alpha', type=float, default=1e-2, help='Regularization parameter for ssc')
    parser.add_argument('--max_iter', type=int, default=200, help='Maximum number of iterations (default: 50)')
    parser.add_argument('--tol', type=float, default=1e-6, help='Tolerance for stopping criterion (default: 1e-6)')
    parser.add_argument('--random_state', type=int, default=None, help='Random seed for clustering (default: None)')
    parser.add_argument('--model', type=str, choices=['sscnmf', 'ricc', 'gnmf', 'gpcanmf', 'dscnmf', 'onmf', 'deepnmf', 'deepsscnmf', 'ssc-omp-nmf'],
                        help='Model to use for clustering')
    parser.add_argument('--graph_weighting', type=str, default='dot-product', choices=['heat-kernel', 'binary', 'dot-product'],)
    parser.add_argument('--l1_reg', type=float, default=0.01,
                        help='L1 regularization parameter for ONMF-ReLU/GPCANMF')
    parser.add_argument('--n_nonzero_coefs', type=int, default=8, help='Number of non-zero coefficients for OMP (default: 8)')
    return parser.parse_args()

def main(model, r, K, sigma=0.0, alpha=0.1, l1_reg=0.01, random_state=None, max_iter=50, tol=1e-6, graph_weighting='dot-product', n_nonzero_coefs=8):
    # load pbmc data
    df = pd.read_csv('data/pbmc_SeqWell_norm_common_sqrt_tp.txt', index_col=0, sep='\t')
    pbmc_values = df.values
    X = pbmc_values.T
    X = X[:-1, :]  # remove last row which is cell name

    # turn labels into numeric labels
    le = LabelEncoder()
    true_labels = le.fit_transform(df.index.str.split('_').str[-1].values)

    np.random.seed(random_state)

    if sigma > 0:
        # Add non-negative Gaussian noise to the data
        noise = np.random.normal(0, sigma, X.shape)
        X += noise
        X = normalize(X, axis=0)
    
    if model == 'sscnmf':
        project_name = 'sscnmf-pbmc'
        acc, ARI, NMI, reconstruction_error, _, _, _ = ssc_nmf_baseline(
            X, K=K, r=r, true_labels=true_labels, alpha=alpha)
    elif model == 'ssc-omp-nmf':
        project_name = 'ssc-omp-nmf-pbmc'
        acc, ARI, NMI, reconstruction_error, _, _, _ = ssc_omp_nmf_baseline(
            X, K=K, r=r, true_labels=true_labels, n_nonzero_coefs=n_nonzero_coefs, random_state=random_state)
    elif model == 'ricc':
        project_name = 'ricc-pbmc'
        acc, ARI, NMI, reconstruction_error, _, _, _ = iter_reg_coneclus_warmstart(
            X, K=K, r=r, true_labels=true_labels,
            alpha=alpha, max_iter=max_iter, NMF_method='anls', ord=2, random_state=random_state)
    elif model == 'gnmf':
        project_name = 'gnmf-pbmc'
        acc, ARI, NMI, reconstruction_error, _, _, _ = GNMF_clus(
            X, K=K, r=r, true_labels=true_labels, weight_type=graph_weighting, max_iter=max_iter)
    elif model == 'gpcanmf':
        project_name = 'gpcanmf-pbmc'
        acc, ARI, NMI, reconstruction_error, _, _, _ = gpca_nmf(
            X, K=K, r=r, true_labels=true_labels,
            l1_reg=l1_reg, random_state=random_state)
    elif model == 'dscnmf':
        project_name = 'dscnmf-pbmc'
        acc, ARI, NMI, reconstruction_error = dsc_nmf_baseline(
            X, K=K, r=r, true_labels=true_labels)
    elif model == 'onmf':
        project_name = 'onmf-pbmc'
        acc, ARI, NMI, reconstruction_error = onmf_ding(
            X, K=K, true_labels=true_labels)
    elif model == 'deepnmf':
        project_name = 'deepnmf-pbmc'
        acc, ARI, NMI, reconstruction_error, _, _ = deep_nmf(
            X, random_state=random_state, true_labels=true_labels)
    elif model == 'deepsscnmf':
        project_name = 'deepsscnmf-pbmc'
        acc, ARI, NMI, reconstruction_error = deep_ssc_nmf(
            X, ranks=[256, 128, 64], alpha=alpha, n_iter=max_iter,
            true_labels=true_labels)
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
    graph_weighting = args.graph_weighting
    n_nonzero_coefs = args.n_nonzero_coefs

    main(model, r, K, sigma, alpha, l1_reg, random_state, max_iter, tol, graph_weighting, n_nonzero_coefs)
    
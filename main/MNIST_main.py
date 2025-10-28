import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
import argparse
import wandb
import torch
from sklearn.metrics import adjusted_rand_score
from src.GenNMF import *
from src.modified_dscnmf import *
from src.baseline import *
from src.deepNMF import *
from src.deepSSCNMF import *
from sklearn.linear_model import Lasso
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from src.nmf import *
from sklearn.datasets import fetch_olivetti_faces, fetch_openml 
from kymatio.torch import Scattering2D
from torchvision import transforms

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
    parser.add_argument('--random_state', type=int, default=None, help='Random seed for clustering (default: None)')
    parser.add_argument('--model', type=str, choices=['sscnmf', 'ricc', 'gnmf', 'gpcanmf', 'onmf_relu', 'dscnmf', 'onmf', 'deepnmf', 'deepsscnmf', 'ssc-omp-nmf'],
    help='Model to use for clustering')
    parser.add_argument('--n_nonzero_coefs', type=int, default=8, help='Number of non-zero coefficients for OMP')
    parser.add_argument('--l1_reg', type=float, default=0.01, help='L1 regularization parameter for ONMF-ReLU/GPCANMF')
    return parser.parse_args()

def main(model, r, n, K, sigma=0.0, alpha = 0.1, l1_reg=0.01, random_state=None, max_iter=500, tol=1e-6, n_nonzero_coefs=8):
    # np.random.seed(random_state)
    # mnist = fetch_openml('mnist_784', version=1)
    # X_full = mnist.data.to_numpy() 
    # y_full = mnist.target.to_numpy().astype(int) 

    # # 2. Subset digits 0-5
    # X_list = []
    # labels = []

    # for digit in range(K):
    #     idx = np.where(y_full == digit)[0]
    #     selected_idx = np.random.choice(idx, n, replace=False)
    #     X_list.append(X_full[selected_idx])
    #     labels.append(np.full(len(selected_idx), digit))

    np.random.seed(random_state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mnist = fetch_openml('mnist_784', version=1)
    X_full = mnist.data.to_numpy().reshape(-1, 28, 28)
    y_full = mnist.target.to_numpy().astype(int)

    X_list, labels = [], []
    for digit in range(K):
        idx = np.where(y_full == digit)[0]
        selected_idx = np.random.choice(idx, n, replace=False)
        X_list.append(X_full[selected_idx])
        labels.append(np.full(len(selected_idx), digit))

    X_subset = np.concatenate(X_list, axis=0)  # shape (K*n, 28, 28)
    true_labels = np.concatenate(labels)

    # -----------------------------
    # 2. Resize to 32×32 (matches reference)
    # -----------------------------
    resize_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    print("Resizing MNIST images to 32×32...")
    X_resized = torch.stack([resize_transform(x.astype(np.uint8)) for x in X_subset])
    # shape: (K*n, 1, 32, 32)

    # -----------------------------
    # 3. Scattering transform (J=3)
    # -----------------------------
    print("Computing scattering on MNIST...")
    scattering = Scattering2D(J=3, shape=(32, 32))

    X_tensor = X_resized.to(device)
    with torch.no_grad():
        scatter_feats = scattering(X_tensor)  # (N, C, H, W)

    # -----------------------------
    # 4. Normalize and flatten
    # -----------------------------
    data = scatter_feats.cpu().numpy()
    data = np.squeeze(data)
    print(data.shape)
    n_sample, C, H, W = data.shape
    data = data.reshape(n_sample, C, -1)

    # Nonnegative normalization for NMF compatibility
    data_min = data.min(axis=2, keepdims=True)
    data_max = data.max(axis=2, keepdims=True)
    data = (data - data_min) / (data_max - data_min + 1e-8)

    # Flatten all transforms
    data = data.reshape(n_sample, -1)

    # -----------------------------
    # 5. Dimensionality reduction (PCA)
    # -----------------------------
    print("Reducing dimensionality with PCA...")
    pca = PCA(n_components=300, random_state=random_state)
    X_reduced = pca.fit_transform(data)

    # -----------------------------
    # 6. Normalize and optionally add nonnegative noise
    # -----------------------------
    X_reduced = normalize(X_reduced, axis=0)

    # # zero truncate
    X_reduced = np.maximum(X_reduced, 0)
    X_reduced = X_reduced.T

    # X_subset = data.T  # shape (features, samples)

    if sigma > 0:
        noise = np.random.normal(0, sigma, X_reduced.shape)
        X_reduced += noise
        X_reduced = np.maximum(X_reduced, 0)  # truncate negatives

    # -----------------------------
    # 7. Final output
    # -----------------------------
    print("Final feature shape:", X_subset.shape)
    print("Labels shape:", true_labels.shape)
    print("Done.")
    if model == 'sscnmf':
        project_name = 'sscnmf-MNIST'
    elif model == 'ssc-omp-nmf':
        project_name = 'ssc-omp-nmf-MNIST'
    elif model == 'ricc':
        project_name = 'ricc-MNIST'
    elif model == 'gnmf':
        project_name = 'gnmf-MNIST'
    elif model == 'gpcanmf':
        project_name = 'gpcanmf-MNIST'
    elif model == 'dscnmf':
        project_name = 'dscnmf-MNIST'
    elif model == 'onmf':
        project_name = 'onmf-MNIST'
    elif model == 'deepnmf':
        project_name = 'deepnmf-MNIST'
    elif model == 'deepsscnmf':
        project_name = 'deepsscnmf-MNIST'

    wandb.init(
        project="coneClustering",
        name=project_name
    )

    if model == 'sscnmf':
        acc, ARI, NMI, reconstruction_error, _, _, _ = ssc_nmf_baseline(
            X_reduced, K, r, true_labels=true_labels, alpha=alpha)
    elif model == 'ssc-omp-nmf':
        acc, ARI, NMI, reconstruction_error, _, _, _ = ssc_omp_nmf_baseline(
            X_reduced, K, r, true_labels=true_labels, n_nonzero_coefs=n_nonzero_coefs, random_state=random_state)
    elif model == 'ricc':
        acc, ARI, NMI, reconstruction_error, _, _, _ = iter_reg_coneclus_warmstart(
            X_reduced, K, r, true_labels=true_labels)
    elif model == 'gnmf':
        acc, ARI, NMI, reconstruction_error, _, _, _ = GNMF_clus(
            X_subset, K=K, r=r, true_labels=true_labels)
    elif model == 'gpcanmf':
        acc, ARI, NMI, reconstruction_error, _, _, _ = gpca_nmf(
            X_subset, K, r, true_labels=true_labels, l1_reg=l1_reg, random_state=random_state)
    elif model == 'dscnmf':
        acc, ARI, NMI, reconstruction_error, _, _ = dsc_nmf_baseline(
            X_subset, K=K, r=r, true_labels=true_labels)
    elif model == 'onmf':
        acc, ARI, NMI, reconstruction_error = onmf_ding(
            X_subset, K=K, true_labels=true_labels, random_state=random_state)
    elif model == 'deepnmf':    
        acc, ARI, NMI, reconstruction_error, _, _ = deep_nmf(
            X_subset, random_state=random_state, true_labels=true_labels)
    elif model == 'deepsscnmf':
        acc, ARI, NMI, reconstruction_error = deep_ssc_nmf(
            X_subset, ranks=[256, 128, 64], alpha=alpha, n_iter=max_iter,
            true_labels=true_labels)

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
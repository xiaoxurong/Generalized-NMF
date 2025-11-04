import numpy as np
import wandb
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import NMF
from src.nmf import *
from sklearn.linear_model import Lasso
from sklearn import cluster
from libnmf.gnmf import GNMF
from libnmf.nmfbase import NMFBase
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph
from scipy.sparse.linalg import svds
from src.utils import *
from src.accuracy import *
from scipy.optimize import nnls, minimize
from sklearn.preprocessing import normalize
from sklearn.linear_model import OrthogonalMatchingPursuit

from src.ssc_omp import *

def data_simulation(m, r, n_k, K, sigma=0.0, random_state=None):
    """
    Simulate K distinct subspaces in R^m, each with dimension r,
    ensuring they differ by generating orthonormal bases.

    Parameters
    ----------
    m : int
        Dimension of the ambient space.
    r : int
        Dimension (rank) of each subspace.
    n_k : int
        Number of points per subspace.
    K : int
        Number of subspaces.
    sigma : float
        Standard deviation of the Gaussian noise to add to each subspace (optional).
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (m, K*n)
        Concatenated data for all subspaces.
    labels : ndarray of shape (K*n,)
        True cluster labels indicating the subspace for each column.
    """
    np.random.seed(random_state)

    X_list = []
    labels_list = []

    # Generate K distinct orthonormal bases
    for k in range(K):
        # Orthonormal basis for the k-th subspace
        U_k = np.abs(np.random.rand(m, r))
        V_k = np.abs(np.random.rand(r, n_k))
        X_k = np.dot(U_k, V_k)
        if sigma > 0:
            noise = np.random.normal(0, sigma, X_k.shape)
            X_k += noise
        # Ensure non-negativity
        X_k = np.maximum(X_k, 0)
        X_list.append(X_k)
        labels_list.append(np.full(n_k, k, dtype=int))

    # Concatenate subspace data (columns) and labels
    X = np.concatenate(X_list, axis=1)       # shape (m, K*n_k)
    labels = np.concatenate(labels_list)     # shape (K*n_k,)

    return X, labels

def baseline_nmf(X, r, max_iter=1000, tol=1e-6, random_state=None):
    np.random.seed(random_state)
    U, V = anls(X, r, max_iter=max_iter, tol=tol)
    X_new = U @ V
    if np.linalg.norm(X) == 0:
        reconstruction_error = np.nan
    else:
        reconstruction_error = np.linalg.norm(X_new - X) / np.linalg.norm(X)
    print(f"Baseline NMF reconstruction error: {reconstruction_error}")
    return U, V, reconstruction_error

def baseline_ksubspace(X, r, K, true_labels, max_iter=1000, tol=1e-6, random_state=None):
    np.random.seed(random_state)
    n = X.shape[1]
    # 1) Initialize cluster labels randomly
    cluster_labels = np.tile(np.arange(K), n//K)
    np.random.shuffle(cluster_labels)  

    iter = 0
    while iter < max_iter:
        sub_datasets = []
        subspace_bases = []
        for k_ in range(K):
            idx_k = np.where(cluster_labels == k_)[0]
            sub_datasets.append(X[:, idx_k])

            U_k, _, _ = np.linalg.svd(sub_datasets[k_], full_matrices=False)
            # zero truncate U_k
            U_k = np.where(U_k > 0, U_k, 0)
            subspace_bases.append(U_k[:, :r])
        
        new_labels = np.zeros_like(cluster_labels)
        for i in range(n):
            x_i = X[:, i]
            best_k = 0
            best_dist = np.inf
            for k_ in range(K):
                U_k = subspace_bases[k_]
                proj_i = U_k @ np.linalg.pinv(U_k.T @ U_k) @ (U_k.T @ x_i)
                dist = np.linalg.norm(x_i - proj_i)
                if dist < best_dist:
                    best_dist = dist
                    best_k = k_
            new_labels[i] = best_k

        num_changed = np.sum(new_labels != cluster_labels)
        if num_changed == 0:
            break
        cluster_labels = new_labels.copy()
        iter += 1

    acc = remap_accuracy(true_labels, cluster_labels)
    ARI = adjusted_rand_score(true_labels, cluster_labels)
    NMI = normalized_mutual_info_score(true_labels, cluster_labels)
    
    return cluster_labels, acc, ARI, NMI

def baseline_ssc(X, true_labels, alpha, random_state=None):
    # X: [number of samples, dimension]
    n_samples = X.shape[1]
    X = X - X.mean(axis=1, keepdims=True)

    C = np.zeros((n_samples, n_samples))

    # Perform Lasso regression for each sample
    for i in range(n_samples):
        x_i = X[:, i]
        X_rest = np.delete(X, i, axis=1)

        # Solve Lasso problem: minimize ||x_i - X_rest * c||^2 + alpha * ||c||_1
        lasso = Lasso(alpha=alpha, fit_intercept=False)#, max_iter=1000)
        lasso.fit(X_rest, x_i)
        c = lasso.coef_

        # Insert c into C matrix
        C[np.arange(n_samples) != i, i] = c
    # 0 truncate C
    C[C < 0] = 0
    # Symmetrize the affinity matrix
    C = 0.5 * (C + C.T)

    # Run spectral clustering on C
    n_clusters = len(np.unique(true_labels))
    spectral = cluster.SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans', random_state=random_state)
    cluster_labels = spectral.fit_predict(C)
    print("length of cluster_labels: ", len(cluster_labels))

    # Return ARI
    acc = remap_accuracy(true_labels, cluster_labels)
    ARI = adjusted_rand_score(true_labels, cluster_labels)
    NMI = normalized_mutual_info_score(true_labels, cluster_labels)

    return cluster_labels, acc, ARI, NMI

def oracle_nmf(X, K, r, true_labels, random_state=None):
    np.random.seed(random_state)
    n = X.shape[1]
    total_error = 0.0
    cluster_num = K

    for k in range(K):
        idx_k = np.where(true_labels == k)[0]
        X_k = X[:, idx_k]
        U_k, V_k, reconstruction_error_k = baseline_nmf(X_k, r, random_state=random_state)
        if reconstruction_error_k is None:
            cluster_num -= 1
            continue

        total_error += reconstruction_error_k

    reconstruction_error = total_error / cluster_num if cluster_num > 0 else np.nan

    return U_k, V_k, reconstruction_error

def ksub_nmf_baseline(X, r, K, true_labels, max_iter=1000, tol=1e-6, random_state=None):
    np.random.seed(random_state)
    n = X.shape[1]
    # 1) Initialize cluster labels randomly
    cluster_labels = np.tile(np.arange(K), n//K)
    np.random.shuffle(cluster_labels)

    # run k-subspace clustering once
    new_labels = np.zeros_like(cluster_labels)
    for i in range(n):
        x_i = X[:, i]
        best_k = 0
        best_dist = np.inf
        for k_ in range(K):
            U_k = np.linalg.svd(X[:, cluster_labels == k_], full_matrices=False)[0]
            U_k = np.where(U_k > 0, U_k, 0)
            proj_i = U_k @ np.linalg.pinv(U_k.T @ U_k) @ (U_k.T @ x_i)
            dist = np.linalg.norm(x_i - proj_i)
            if dist < best_dist:
                best_dist = dist
                best_k = k_
        new_labels[i] = best_k

    cluster_labels = new_labels.copy()

    acc = remap_accuracy(true_labels, cluster_labels)
    ARI = adjusted_rand_score(true_labels, cluster_labels)
    NMI = normalized_mutual_info_score(true_labels, cluster_labels)

    sub_datasets = []
    subspace_bases = []
    X_new = np.zeros_like(X)
    for k_ in range(K):
        idx_k = np.where(new_labels == k_)[0]
        sub_datasets.append(X[:, idx_k])

        if len(sub_datasets[k_]) == 0:
            subspace_bases.append(None)
            continue
        else:
            U_k = NMF(n_components=r, random_state=random_state, max_iter=1000).fit_transform(sub_datasets[k_])
            U_k = np.where(U_k > 0, U_k, 0)
            subspace_bases.append(U_k[:, :r])

        X_k = sub_datasets[k_]
        X_new[:, idx_k] = U_k @ np.linalg.pinv(U_k.T @ U_k) @ (U_k.T @ X_k)
    # calculate reconstruction error
    reconstruction_error = np.linalg.norm(X_new - X) / np.linalg.norm(X)

    return reconstruction_error, acc, ARI, NMI

def ssc_nmf_baseline(X, r, K, true_labels, max_iter=1000, random_state=None, alpha=0.01):
    np.random.seed(random_state)

    # Step 1: SSC clustering
    pred_labels, acc, ARI, NMI = baseline_ssc(X, true_labels, alpha, random_state=random_state)

    # # Step 2: Initialize containers
    # sub_datasets = []
    subspace_bases = []
    subspace_coef = []
    
    X_new = np.zeros_like(X)

    # Step 3: Per-cluster NMF
    for k_ in range(K):
        idx_k = np.where(pred_labels == k_)[0]
        X_k = X[:, idx_k]
        # sub_datasets.append(X_k)

        if X_k.shape[1] == 0:
            subspace_bases.append(None)
            continue

        # Fit NMF and store basis
        model = NMF(n_components=r, random_state=random_state, max_iter=max_iter)
        U_k = model.fit_transform(X_k)
        # U_k = np.maximum(U_k, 0)  # optional ReLU
        subspace_bases.append(U_k)
        H_k = model.components_
        subspace_coef.append(H_k)

        # Project back into subspace
        X_new[:, idx_k] = U_k @ H_k


    # Step 4: Evaluate reconstruction error
    reconstruction_error = np.linalg.norm(X_new - X) / np.linalg.norm(X)

    return acc, ARI, NMI, reconstruction_error, pred_labels, subspace_bases, subspace_coef

def ssc_omp_nmf_baseline(X, r, K, true_labels, max_iter=1000, random_state=None, n_nonzero_coefs=8):
    np.random.seed(random_state)

    # Step 1: SSC clustering
    # use the ssc_omp function to do the clustering
    model = SparseSubspaceClusteringOMP(n_clusters=K, random_state=random_state)
    print("Fitting SSC-OMP model... on data with shape:", X.T.shape)
    model.fit(X.T)
    pred_labels = model.labels_

    # # Step 2: Initialize containers
    # sub_datasets = []
    subspace_bases = []
    subspace_coef = []
    
    X_new = np.zeros_like(X)

    # Step 3: Per-cluster NMF
    for k_ in range(K):
        idx_k = np.where(pred_labels == k_)[0]
        X_k = X[:, idx_k]

        if X_k.shape[1] == 0:
            subspace_bases.append(None)
            continue

        # Fit NMF and store basis
        model = NMF(n_components=r, random_state=random_state, max_iter=max_iter)
        U_k = model.fit_transform(X_k)
        # U_k = np.maximum(U_k, 0)  # optional ReLU
        subspace_bases.append(U_k)
        H_k = model.components_
        subspace_coef.append(H_k)

        # Project back into subspace
        X_new[:, idx_k] = U_k @ H_k


    # Step 4: Evaluate reconstruction error
    reconstruction_error = np.linalg.norm(X_new - X) / np.linalg.norm(X)
    # acc = remap_accuracy(true_labels, pred_labels)
    acc = clustering_accuracy(true_labels, pred_labels)
    ARI = adjusted_rand_score(true_labels, pred_labels)
    NMI = normalized_mutual_info_score(true_labels, pred_labels)

    return acc, ARI, NMI, reconstruction_error, pred_labels, subspace_bases, subspace_coef

def GNMF_clus(X, K, true_labels, max_iter=1000, random_state=None, lmd=0.1, weight_type='heat-kernel', param=0.3):
    """
    Graph-based Non-negative Matrix Factorization (GNMF) for subspace clustering.
    """
    # base = NMFBase(X, K)
    model = GNMF(X, K)
    model.compute_factors(max_iter=max_iter, lmd=lmd, weight_type=weight_type, param=param)

    W = model.W
    H = model.H

    H_array = np.asarray(H)
    predicted_labels = KMeans(n_clusters=K, random_state=random_state).fit_predict(H_array.T)

    acc = remap_accuracy(true_labels, predicted_labels)
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    reconstruction_error = np.linalg.norm(X - W @ H) / np.linalg.norm(X)

    return acc, ari, nmi, reconstruction_error

def iter_reg_coneclus_warmstart(X, K, r, true_labels, max_iter=1000, random_state=None,
                                NMF_method='anls', alpha=0.01, ord=2):
    np.random.seed(random_state)
    n = X.shape[1]
    m = X.shape[0]

    # 1. Random init of cluster labels
    cluster_labels = np.tile(np.arange(K), n // K + 1)[:n]
    np.random.shuffle(cluster_labels)

    # 2. Warm start: track old cluster assignment
    prev_labels = np.full(n, -1)
    
    # 3. Store W, H for each cluster
    nmf_bases = [None] * K
    nmf_components = [None] * K

    model_template = lambda: NMF(n_components=r, init='random', random_state=random_state,
                                 max_iter=1000, solver='cd')

    for iteration in range(max_iter):
        # Flag to check if cluster content changed
        cluster_changed = [False] * K

        # 4. Partition data by clusters
        sub_datasets = [X[:, cluster_labels == k_] for k_ in range(K)]

        # 5. NMF on each cluster with warm start
        for k_ in range(K):
            x_k = sub_datasets[k_]
            if x_k.shape[1] == 0:
                nmf_bases[k_] = None
                nmf_components[k_] = None
                continue

            # Check if cluster content changed
            idx = (cluster_labels == k_)
            cluster_changed[k_] = not np.array_equal(prev_labels[idx], cluster_labels[idx])

            if not cluster_changed[k_] and nmf_bases[k_] is not None:
                # Skip recomputing if cluster didn't change
                continue

            model = model_template()
            x_k = np.maximum(x_k, 0)

            if nmf_bases[k_] is not None and nmf_components[k_] is not None:
                # Warm start
                try:
                    U_k = model.fit_transform(x_k, W=nmf_bases[k_], H=nmf_components[k_])
                except:
                    U_k = model.fit_transform(x_k)  # fallback
            else:
                U_k = model.fit_transform(x_k)

            V_k = model.components_
            nmf_bases[k_] = U_k
            nmf_components[k_] = V_k

        # 6. Reassign labels using projection distances
        all_dists = np.full((K, n), np.inf)

        for k_ in range(K):
            U_k = nmf_bases[k_]
            if U_k is None or U_k.shape[1] == 0:
                continue
            try:
                # Precompute (UᵀU)^(-1) UᵀX for efficiency
                C_k, _, _, _ = np.linalg.lstsq(U_k, X, rcond=None)
                C_k_relu = np.where(C_k > 0, C_k, 0)
                X_proj_k = U_k @ C_k_relu
                dist_k = np.linalg.norm(X - X_proj_k, axis=0) + alpha * np.linalg.norm(C_k_relu, ord=ord, axis=0)
                all_dists[k_] = dist_k
            except np.linalg.LinAlgError:
                continue

        new_labels = np.argmin(all_dists, axis=0)
        num_changed = np.sum(new_labels != cluster_labels)

        if num_changed == 0:
            break

        prev_labels = cluster_labels.copy()
        cluster_labels = new_labels.copy()

    # 7. Final reconstruction (optional: reuse warm-start W, H)
    X_reconstructed = np.zeros_like(X)
    cluster_Hs = [None] * K
    for k_ in range(K):
        idx_k = np.where(cluster_labels == k_)[0]
        if len(idx_k) == 0:
            continue
        x_k = np.maximum(X[:, idx_k], 0)
        model = model_template()
        U_k = model.fit_transform(x_k)
        V_k = model.components_
        nmf_bases[k_] = U_k
        nmf_components[k_] = V_k
        cluster_Hs[k_] = V_k
        X_reconstructed[:, idx_k] = U_k @ V_k

    # 8. Evaluation
    reconstruction_error = np.linalg.norm(X_reconstructed - X) / np.linalg.norm(X)
    proportion_negatives = np.sum(X_reconstructed < 0) / X_reconstructed.size
    acc = remap_accuracy(true_labels, cluster_labels)
    ARI = adjusted_rand_score(true_labels, cluster_labels)
    NMI = normalized_mutual_info_score(true_labels, cluster_labels)

    return acc, ARI, NMI, reconstruction_error, cluster_labels, nmf_bases, nmf_components

def iter_ssc_nmf(X, K, r, true_labels, max_iter=50, random_state=None,
                 alpha_ssc=0.01, l1_reg=0.1, use_sparse=True):
    np.random.seed(random_state)
    n = X.shape[1]
    m = X.shape[0]

    cluster_labels = ssc_func(X.T, K, alpha=alpha_ssc)
    prev_labels = np.full(n, -1)

    nmf_bases = [None] * K
    nmf_components = [None] * K

    def run_nmf_block(X_block, r, W_init=None, H_init=None):
        if use_sparse:
            return sparse_nmf(X_block, r, l1_reg=l1_reg, W_init=W_init, H_init=H_init)
        else:
            model = NMF(n_components=r, init='random', random_state=random_state,
                               max_iter=500)
            W = model.fit_transform(X_block)
            H = model.components_
            return W, H

    for iteration in range(max_iter):
        cluster_changed = [False] * K
        sub_datasets = [X[:, cluster_labels == k_] for k_ in range(K)]

        for k_ in range(K):
            x_k = sub_datasets[k_]
            if x_k.shape[1] == 0:
                nmf_bases[k_] = None
                nmf_components[k_] = None
                continue

            idx = (cluster_labels == k_)
            cluster_changed[k_] = not np.array_equal(prev_labels[idx], cluster_labels[idx])
            if not cluster_changed[k_] and nmf_bases[k_] is not None:
                continue

            x_k = np.maximum(x_k, 0)
            W_init = nmf_bases[k_]
            H_init = nmf_components[k_]
            if H_init is not None and H_init.shape[1] != x_k.shape[1]:
                H_init = None

            Wk, Hk = run_nmf_block(x_k, r, W_init=W_init, H_init=H_init)
            nmf_bases[k_] = Wk
            nmf_components[k_] = Hk

        # Recluster using SSC on fresh NMF latent features
        X_latent = np.zeros((r, n))
        for k_ in range(K):
            idx_k = np.where(cluster_labels == k_)[0]
            if len(idx_k) == 0:
                continue
            x_k = X[:, idx_k]
            x_k = np.maximum(x_k, 0)
            _, Hk = run_nmf_block(x_k, r)
            if Hk.shape[1] != len(idx_k):
                continue
            X_latent[:, idx_k] = Hk

        new_labels = ssc_func(X_latent.T, K, alpha=alpha_ssc)
        num_changed = np.sum(new_labels != cluster_labels)
        if num_changed == 0:
            break

        prev_labels = cluster_labels.copy()
        cluster_labels = new_labels.copy()

    # Final reconstruction
    X_reconstructed = np.zeros_like(X)
    for k_ in range(K):
        idx_k = np.where(cluster_labels == k_)[0]
        if len(idx_k) == 0:
            continue
        x_k = X[:, idx_k]
        x_k = np.maximum(x_k, 0)
        W_init = nmf_bases[k_]
        H_init = nmf_components[k_]
        if H_init is not None and H_init.shape[1] != x_k.shape[1]:
            H_init = None

        Wk, Hk = run_nmf_block(x_k, r, W_init=W_init, H_init=H_init)
        X_reconstructed[:, idx_k] = Wk @ Hk

    reconstruction_error = np.linalg.norm(X_reconstructed - X) / np.linalg.norm(X)
    proportion_negatives = np.sum(X_reconstructed < 0) / X_reconstructed.size
    acc = remap_accuracy(true_labels, cluster_labels)
    ARI = adjusted_rand_score(true_labels, cluster_labels)
    NMI = normalized_mutual_info_score(true_labels, cluster_labels)

    return acc, ARI, NMI, reconstruction_error, proportion_negatives

def ssc_sparse_nmf_baseline(X, r, K, true_labels, max_iter=1000, random_state=None, alpha=0.01, l1_reg=0.1):
    np.random.seed(random_state)
    
    # 1. SSC clustering
    pred_labels, acc, ARI, NMI = baseline_ssc(X, true_labels, alpha=alpha)

    sub_datasets = []
    subspace_bases = []
    X_reconstructed = np.zeros_like(X)

    for k_ in range(K):
        idx_k = np.where(pred_labels == k_)[0]
        X_k = X[:, idx_k]
        sub_datasets.append(X_k)

        if X_k.shape[1] == 0:
            subspace_bases.append(None)
            continue

        # 2. Sparse NMF on each cluster
        W, H = sparse_nmf(X_k, r=r, l1_reg=l1_reg)
        subspace_bases.append(W)

        # 3. Reconstruct each cluster
        X_reconstructed[:, idx_k] = W @ H

    # 4. Compute reconstruction error
    reconstruction_error = np.linalg.norm(X_reconstructed - X) / np.linalg.norm(X)

    return acc, ARI, NMI, reconstruction_error

def iter_reg_coneclus_sparse_nmf(X, K, r, true_labels, max_iter=50, random_state=None,
                                 alpha=0.01, ord=2, l1_reg=0.1):
    np.random.seed(random_state)
    n = X.shape[1]
    m = X.shape[0]

    # 1. Random init of cluster labels
    cluster_labels = np.tile(np.arange(K), n // K + 1)[:n]
    np.random.shuffle(cluster_labels)

    # 2. Warm start: track old cluster assignment
    prev_labels = np.full(n, -1)
    
    # 3. Store W, H for each cluster
    nmf_bases = [None] * K
    nmf_components = [None] * K

    def sparse_nmf(X_block, r, W_init=None, H_init=None, n_iter=200):
        W = np.random.rand(X_block.shape[0], r) if W_init is None else W_init
        H = np.random.rand(r, X_block.shape[1]) if H_init is None else H_init
        for _ in range(n_iter):
            WH = W @ H
            H *= (W.T @ X_block) / (W.T @ WH + l1_reg + 1e-10)
            W *= (X_block @ H.T) / (W @ (H @ H.T) + 1e-10)
        return W, H

    for iteration in range(max_iter):
        cluster_changed = [False] * K
        sub_datasets = [X[:, cluster_labels == k_] for k_ in range(K)]

        for k_ in range(K):
            x_k = sub_datasets[k_]
            if x_k.shape[1] == 0:
                nmf_bases[k_] = None
                nmf_components[k_] = None
                continue

            idx = (cluster_labels == k_)
            cluster_changed[k_] = not np.array_equal(prev_labels[idx], cluster_labels[idx])
            if not cluster_changed[k_] and nmf_bases[k_] is not None:
                continue

            x_k = np.maximum(x_k, 0)
            W_init = nmf_bases[k_]
            H_init = nmf_components[k_]
            if H_init is not None and H_init.shape[1] != x_k.shape[1]:
                H_init = None

            Wk, Hk = sparse_nmf(x_k, r=r, W_init=W_init, H_init=H_init, n_iter=200)
            nmf_bases[k_] = Wk
            nmf_components[k_] = Hk

        # 6. Reassign labels using projection distances
        all_dists = np.full((K, n), np.inf)

        for k_ in range(K):
            U_k = nmf_bases[k_]
            if U_k is None or U_k.shape[1] == 0:
                continue
            try:
                # Project entire X onto each U_k
                C_k, _, _, _ = np.linalg.lstsq(U_k, X, rcond=None)
                C_k_relu = np.maximum(C_k, 0)
                X_proj_k = U_k @ C_k_relu
                dist_k = np.linalg.norm(X - X_proj_k, axis=0) + alpha * np.linalg.norm(C_k_relu, ord=ord, axis=0)
                all_dists[k_] = dist_k
            except np.linalg.LinAlgError:
                continue

        new_labels = np.argmin(all_dists, axis=0)
        num_changed = np.sum(new_labels != cluster_labels)
        if num_changed == 0:
            break

        prev_labels = cluster_labels.copy()
        cluster_labels = new_labels.copy()

    # 7. Final reconstruction
    X_reconstructed = np.zeros_like(X)
    for k_ in range(K):
        idx_k = np.where(cluster_labels == k_)[0]
        if len(idx_k) == 0:
            continue
        x_k = X[:, idx_k]
        x_k = np.maximum(x_k, 0)
        Wk, Hk = sparse_nmf(x_k, r=r, n_iter=200)
        X_reconstructed[:, idx_k] = Wk @ Hk

    # 8. Evaluation
    reconstruction_error = np.linalg.norm(X_reconstructed - X) / np.linalg.norm(X)
    proportion_negatives = np.sum(X_reconstructed < 0) / X_reconstructed.size
    acc = remap_accuracy(true_labels, cluster_labels)
    ARI = adjusted_rand_score(true_labels, cluster_labels)
    NMI = normalized_mutual_info_score(true_labels, cluster_labels)

    return acc, ARI, NMI, reconstruction_error, proportion_negatives

def gpca_nmf(X, K, r, true_labels, use_sparse=False, l1_reg=0.1, random_state=None):
    """
    GPCA + NMF pipeline using sklearn-friendly tools.
    """
    np.random.seed(random_state)
    n = X.shape[1]
    X_reconstructed = np.zeros_like(X)

    # Step 1: GPCA clustering
    pred_labels = approximate_gpca(X, K, random_state=random_state)

    # Step 2: Cluster-wise NMF
    for k in range(K):
        idx_k = np.where(pred_labels == k)[0]
        X_k = X[:, idx_k]
        if X_k.shape[1] == 0:
            continue
        X_k = np.maximum(X_k, 0)

        if use_sparse:
            W = np.random.rand(X_k.shape[0], r)
            H = np.random.rand(r, X_k.shape[1])
            for _ in range(200):
                WH = W @ H
                H *= (W.T @ X_k) / (W.T @ WH + l1_reg + 1e-10)
                W *= (X_k @ H.T) / (W @ (H @ H.T) + 1e-10)
        else:
            model = SklearnNMF(n_components=r, init='random', max_iter=1000, random_state=random_state)
            W = model.fit_transform(X_k)
            H = model.components_

        X_reconstructed[:, idx_k] = W @ H

    # Step 3: Evaluation
    acc = remap_accuracy(true_labels, pred_labels)
    ARI = adjusted_rand_score(true_labels, pred_labels)
    NMI = normalized_mutual_info_score(true_labels, pred_labels)
    reconstruction_error = np.linalg.norm(X_reconstructed - X) / np.linalg.norm(X)

    return acc, ARI, NMI, reconstruction_error, pred_labels, W, H

def ssc_projnmf(X, K, r, true_labels, alpha=0.01, max_iter=500):
    """
    SSC + Projective NMF pipeline for representation learning.
    
    Parameters:
        X: (d, n) data matrix
        K: number of clusters
        r: NMF rank
        true_labels: ground-truth labels (n,)
        max_iter: iterations for projective NMF
    
    Returns:
        acc: clustering accuracy from SSC
        ARI: adjusted Rand index from SSC
        NMI: normalized mutual info from SSC
        reconstruction_error: ||X - \hat{X}|| / ||X||
    """
    n = X.shape[1]
    X_reconstructed = np.zeros_like(X)

    # Step 1: SSC clustering
    pred_labels, acc, ARI, NMI = baseline_ssc(X, true_labels, alpha=alpha)

    # Step 2: Cluster-wise Projective NMF
    for k in range(K):
        idx_k = np.where(pred_labels == k)[0]
        X_k = X[:, idx_k]
        if X_k.shape[1] == 0:
            continue

        W_k, loss_k = projective_nmf_orthogonal(X_k, r=r, max_iter=max_iter)
        X_reconstructed[:, idx_k] = W_k @ W_k.T @ X_k

    # Step 3: Evaluate
    reconstruction_error = np.linalg.norm(X - X_reconstructed) / np.linalg.norm(X)

    print("Norm of X:", np.linalg.norm(X))
    print("Norm of X_reconstructed:", np.linalg.norm(X_reconstructed))
    print("Absolute error:", np.linalg.norm(X - X_reconstructed))
    print("Relative error:", np.linalg.norm(X - X_reconstructed) / np.linalg.norm(X))

    return acc, ARI, NMI, reconstruction_error
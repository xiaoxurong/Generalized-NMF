import warnings
import math
import numpy as np
import time
import os
from tqdm import tqdm

from scipy import sparse

from sklearn import cluster
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.decomposition import sparse_encode
from sklearn.linear_model import orthogonal_mp
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state, check_array, check_symmetric

class SelfRepresentation(BaseEstimator, ClusterMixin):
    """Base class for self-representation based subspace clustering.

    Parameters
    -----------
    n_clusters : integer, optional, default: 8
        Number of clusters in the dataset.
    affinity : string, optional, 'symmetrize' or 'nearest_neighbors', default 'symmetrize'
        The strategy for constructing affinity_matrix_ from representation_matrix_.
        If ``symmetrize``, then affinity_matrix_ is set to be
    		|representation_matrix_| + |representation_matrix_|^T.
		If ``nearest_neighbors``, then the affinity_matrix_ is the k nearest
		    neighbor graph for the rows of representation_matrix_
    random_state : int, RandomState instance or None, optional, default: None
        This is the random_state parameter for k-means. 
    n_init : int, optional, default: 10
        This is the n_init parameter for k-means. 
    n_jobs : int, optional, default: 1
        The number of parallel jobs to run.
        If ``-1``, then the number of jobs is set to the number of CPU cores.

    Attributes
    ----------
    representation_matrix_ : array-like, shape (n_samples, n_samples)
        Self-representation matrix. Available only if after calling
        ``fit`` or ``fit_self_representation``.
    labels_ :
        Labels of each point. Available only if after calling ``fit``.
    """

    def __init__(self, n_clusters=8, affinity='symmetrize', random_state=None, n_init=20, n_jobs=1):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.random_state = random_state
        self.n_init = n_init
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Compute representation matrix, then apply spectral clustering
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
        """
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'], dtype=np.float64)
        time_base = time.time()
        
        self._self_representation(X)
        self.timer_self_representation_ = time.time() - time_base
        
        self._representation_to_affinity()
        self._spectral_clustering()
        self.timer_time_ = time.time() - time_base

        return self
	
    def fit_self_representation(self, X, y=None):
        """Compute representation matrix without apply spectral clustering.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
        """
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'], dtype=np.float64)
        time_base = time.time()
        
        self._self_representation(X)
        self.timer_self_representation_ = time.time() - time_base
        
        return self

    def _representation_to_affinity(self):
        """Compute affinity matrix from representation matrix.
        """
        normalized_representation_matrix_ = normalize(self.representation_matrix_, 'l2')
        if self.affinity == 'symmetrize':
            self.affinity_matrix_ = 0.5 * (np.absolute(normalized_representation_matrix_) + np.absolute(normalized_representation_matrix_.T))
        elif self.affinity == 'nearest_neighbors':
            neighbors_graph = kneighbors_graph(normalized_representation_matrix_, 3, 
		                                       mode='connectivity', include_self=False)
            self.affinity_matrix_ = 0.5 * (neighbors_graph + neighbors_graph.T)

    def _spectral_clustering(self):
        affinity_matrix_ = check_symmetric(self.affinity_matrix_)
        random_state = check_random_state(self.random_state)
        
        laplacian = sparse.csgraph.laplacian(affinity_matrix_, normed=True)
        _, vec = sparse.linalg.eigsh(sparse.identity(laplacian.shape[0]) - laplacian, 
                                     k=self.n_clusters, sigma=None, which='LA')
        embedding = normalize(vec)
        _, self.labels_, _ = cluster.k_means(embedding, self.n_clusters, 
                                             random_state=random_state, n_init=self.n_init)

def sparse_subspace_clustering_orthogonal_matching_pursuit(X, n_nonzero=10, thr=1.0e-6, cache_dir="cache"):
    """Sparse subspace clustering by orthogonal matching pursuit (SSC-OMP)
    Compute self-representation matrix C by solving the following optimization problem
    min_{c_j} ||x_j - c_j X ||_2^2 s.t. c_jj = 0, ||c_j||_0 <= n_nonzero
    via OMP, where c_j and x_j are the j-th rows of C and X, respectively

    Parameters
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data to be clustered
    n_nonzero : int, default 10
        Termination condition for omp.
    thr : float, default 1.0e-5
        Termination condition for omp.	

    Returns
    -------
    representation_matrix_ : csr matrix, shape: n_samples by n_samples
        The self-representation matrix.
	
    References
    -----------			
    C. You, D. Robinson, R. Vidal, Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016
    """	
    """SSC-OMP with disk caching."""
    os.makedirs(cache_dir, exist_ok=True)
    # Build a unique cache key
    key = f"ssc_omp_n{n_nonzero}_thr{thr:.0e}_n{X.shape[0]}_{X.shape[1]}.npz"
    cache_path = os.path.join(cache_dir, key)

    # Load cached result if available
    if os.path.exists(cache_path):
        print(f"Loading cached SSC-OMP result: {cache_path}")
        return sparse.load_npz(cache_path)
    n_samples = X.shape[0]
    rows = np.zeros(n_samples * n_nonzero, dtype = int)
    cols = np.zeros(n_samples * n_nonzero, dtype = int)
    vals = np.zeros(n_samples * n_nonzero)
    curr_pos = 0

    for i in tqdm(range(n_samples), desc="Processing samples"):
    # for i in range(n_samples):
        residual = X[i, :].copy()  # initialize residual
        supp = np.empty(shape=(0), dtype = int)  # initialize support
        residual_norm_thr = np.linalg.norm(X[i, :]) * thr
        for t in range(n_nonzero):  # for each iteration of OMP  
            # compute coherence between residuals and X     
            coherence = abs( np.matmul(residual, X.T) )
            coherence[i] = 0.0
            # update support
            supp = np.append(supp, np.argmax(coherence))
            # compute coefficients
            c = np.linalg.lstsq( X[supp, :].T, X[i, :].T, rcond=None)[0]
            # compute residual
            residual = X[i, :] - np.matmul(c.T, X[supp, :])
            # check termination
            if np.sum(residual **2) < residual_norm_thr:
                break

        rows[curr_pos:curr_pos + len(supp)] = i
        cols[curr_pos:curr_pos + len(supp)] = supp
        vals[curr_pos:curr_pos + len(supp)] = c
        curr_pos += len(supp)

#   affinity = sparse.csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples)) + sparse.csr_matrix((vals, (cols, rows)), shape=(n_samples, n_samples))
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))
					

class SparseSubspaceClusteringOMP(SelfRepresentation):
    """Sparse subspace clustering by orthogonal matching pursuit (SSC-OMP). 
    This is a self-representation based subspace clustering method that computes
    the self-representation matrix C via solving the following problem
    min_{c_j} ||x_j - c_j X ||_2^2 s.t. c_jj = 0, ||c_j||_0 <= n_nonzero
    via OMP, where c_j and x_j are the j-th rows of C and X, respectively

    Parameters
    -----------
    n_clusters : integer, optional, default: 8
        Number of clusters in the dataset.
    affinity : string, optional, 'symmetrize' or 'nearest_neighbors', default 'symmetrize'
        The strategy for constructing affinity_matrix_ from representation_matrix_.
    random_state : int, RandomState instance or None, optional, default: None
        This is the random_state parameter for k-means. 
    n_init : int, optional, default: 10
        This is the n_init parameter for k-means. 
    n_nonzero : int, default 10
        Termination condition for omp.
    thr : float, default 1.0e-5
        Termination condition for omp.	
	
    Attributes
    ----------
    representation_matrix_ : array-like, shape (n_samples, n_samples)
        Self-representation matrix. Available only if after calling
        ``fit`` or ``fit_self_representation``.
    labels_ :
        Labels of each point. Available only if after calling ``fit``

    References
    -----------	
    C. You, D. Robinson, R. Vidal, Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016
    """
    def __init__(self, n_clusters=8, affinity='symmetrize', random_state=None, n_init=10, n_jobs=1, n_nonzero=10, thr=1.0e-6):
        self.n_nonzero = n_nonzero
        self.thr = thr
        SelfRepresentation.__init__(self, n_clusters, affinity, random_state, n_init, n_jobs)
    
    def _self_representation(self, X):
        self.representation_matrix_ = sparse_subspace_clustering_orthogonal_matching_pursuit(X, self.n_nonzero, self.thr)
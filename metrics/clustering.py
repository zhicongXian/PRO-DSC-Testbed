"""
from https://github.com/LeslieTrue/CPP
"""
import numpy as np
import sklearn
import sklearn.manifold
import sklearn.cluster
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics.cluster import _supervised
from scipy.optimize import linear_sum_assignment
import scipy.sparse
import torch
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score


def feature_detection(A, label):
    """ Computes proportion of l_1 norm
    of each row of A that is given to connections outside of cluster
    """
    n = A.shape[0]
    err = 0
    # TODO: get rid of for loop
    for i in range(n):
        err += np.abs(A[i,label==label[i]]).sum()/np.abs(A[i,:]).sum()
    err /= n
    err = 1-err
    return err

def nmi(label, pred_label):
    return normalized_mutual_info_score(label, pred_label)

def percent_wrong_edge(A, label):
    row, col = A.nonzero()
    matches = label[row] != label[col]
    if isinstance(matches, torch.Tensor):
        matches = matches.float()
    return matches.mean()*100

def clustering_accuracy(label, pred_label):
    """ from https://github.com/ChongYou/subspace-clustering
    """
    label, pred_label = _supervised.check_clusterings(label, pred_label)
    value = _supervised.contingency_matrix(label, pred_label)
    [r, c] = linear_sum_assignment(-value)
    return value[r, c].sum() / len(label)

def sparsity(A, zero_cutoff=1e-8):
    """ Average number of nonzeros per row
    """
    return np.sum(np.abs(A) > zero_cutoff)/A.shape[0]
def basic_metrics(A, label, verbose=True):
    nnz = sparsity(A)
    fd_error = feature_detection(A, label)
    components = scipy.sparse.csgraph.connected_components(A, return_labels=False)
    wrong_edge = percent_wrong_edge(A, label)
    if verbose:
        print(f"NNZ/ row: {nnz:.2f}   ||| Feat detect: {fd_error:.5f} ")
        print(f"Num comp: {components}       ||| Pct wrong edges: {wrong_edge:.2f}")
    return nnz, fd_error, components, wrong_edge

def spectral_clustering_metrics(A, nclass, label, verbose=True, n_init=10, normalize_embed=True, solver_type='lm', extra_dim=0, tol=0):
    """ n_init is number of separate runs of kmeans to average over
    computes average accuracy and nmi
    """
    lap = scipy.sparse.csgraph.laplacian(A, normed=True)
    # nnz, fd_error, components, wrong_edge = basic_metrics(A, label, verbose=False)
    # if components > nclass:
    #     print('---Oversegmented graph, setting higher eigensolver tolerance (unstable results)---')
    #     # oversegmented, need higher tolerance
    #     tol = 1e-4

    if solver_type=='shift_invert':
        vals, embedding = scipy.sparse.linalg.eigsh(lap, k=nclass+extra_dim, sigma=1e-6, which='LM', tol=tol)
    elif solver_type=='la':
        vals, embedding = scipy.sparse.linalg.eigsh(-lap, k=nclass+extra_dim,
                                    sigma=None,  which='LA', tol=tol)
    elif solver_type=='lm':
        k = nclass+extra_dim

        vals, embedding = scipy.sparse.linalg.eigsh(
            2*scipy.sparse.identity(lap.shape[0])-lap, ncv = max(2 * k + 1, 50),
            k=nclass+extra_dim, sigma=None,  which='LM', tol=tol)
    else:
        raise ValueError('invalid solver')

    if normalize_embed:
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    cluster_model = sklearn.cluster.KMeans(n_clusters=nclass, n_init=n_init) ##--TODO there is also a seed here to be set!!
    acc_lst = []
    nmi_lst = []
    pred_lst = []
    for _ in range(n_init):
        cluster_model.fit(embedding)
        pred_label = cluster_model.labels_
        acc = clustering_accuracy(label, pred_label)
        nmi_score = nmi(label, pred_label)
        acc_lst.append(acc)
        nmi_lst.append(nmi_score)
        pred_lst.append(pred_label)

    #conn_lst = connectivity_lst(A, label)

    if verbose:
        print(f'Acc mean: {np.mean(acc_lst):.3f}   ||| stdev: {np.std(acc_lst):.4f}')
    # if components > nclass:
    #     # do not record unstable results for oversegmented case
    #     acc_lst = [0]

    return acc_lst, nmi_lst, pred_lst # fd_error, nnz


def self_representation_loss(labels_true, representation_matrix):
    """Evaluation of self-representation error for self-expressive subspace clustering methods
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
    	A clustering of the data into disjoint subsets.
    representation_matrix : array, shape = [n_samples, n_samples]
    	Each row is a representation vector

    Returns
    -------
    loss : float
       return self_representation_loss in the range of [0, 1]
    """
    n_samples = labels_true.shape[0]
    loss = 0.0
    for i in range(n_samples):
        representation_vec = np.abs(representation_matrix[i, :])
        label = labels_true[i]
        loss += np.sum(representation_vec[labels_true != label]) / np.sum(representation_vec)

    return loss / n_samples



def spectral_clustering_metrics_with_ari(A, nclass, label, verbose=True, n_init=10, normalize_embed=True, solver_type='lm',
                                extra_dim=0, tol=0, seed = 10):
    """ n_init is number of separate runs of kmeans to average over
    computes average accuracy and nmi
    """
    lap = scipy.sparse.csgraph.laplacian(A, normed=True)
    # nnz, fd_error, components, wrong_edge = basic_metrics(A, label, verbose=False)
    # if components > nclass:
    #     print('---Oversegmented graph, setting higher eigensolver tolerance (unstable results)---')
    #     # oversegmented, need higher tolerance
    #     tol = 1e-4

    if solver_type == 'shift_invert':
        vals, embedding = scipy.sparse.linalg.eigsh(lap, k=nclass + extra_dim, sigma=1e-6, which='LM', tol=tol)
    elif solver_type == 'la':
        vals, embedding = scipy.sparse.linalg.eigsh(-lap, k=nclass + extra_dim,
                                                    sigma=None, which='LA', tol=tol)
    elif solver_type == 'lm':
        k = nclass + extra_dim

        vals, embedding = scipy.sparse.linalg.eigsh(
            2 * scipy.sparse.identity(lap.shape[0]) - lap, ncv=max(2 * k + 1, 50),
            k=nclass + extra_dim, sigma=None, which='LM', tol=tol)
    else:
        raise ValueError('invalid solver')

    if normalize_embed:
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    cluster_model = sklearn.cluster.KMeans(n_clusters=nclass, n_init=n_init, random_state=seed)
    acc_lst = []
    nmi_lst = []
    pred_lst = []
    ari_lst = []
    for _ in range(n_init):
        cluster_model.fit(embedding)
        pred_label = cluster_model.labels_
        acc = clustering_accuracy(label, pred_label)
        nmi_score = nmi(label, pred_label)
        ari = adjusted_rand_score(label, pred_label)
        acc_lst.append(acc)
        nmi_lst.append(nmi_score)
        pred_lst.append(pred_label)
        ari_lst.append(ari)

    # conn_lst = connectivity_lst(A, label)

    if verbose:
        print(f'Acc mean: {np.mean(acc_lst):.3f}   ||| stdev: {np.std(acc_lst):.4f}')
    # if components > nclass:
    #     # do not record unstable results for oversegmented case
    #     acc_lst = [0]

    return acc_lst, nmi_lst, pred_lst, ari_lst   # fd_error, nnz

def spectral_clustering_metrics_with_ari_and_subspace_discovery_error(A, nclass, label, verbose=True, n_init=10, normalize_embed=True, solver_type='lm',
                                extra_dim=0, tol=0, seed = 10):
    """ n_init is number of separate runs of kmeans to average over
    computes average accuracy and nmi
    """
    lap = scipy.sparse.csgraph.laplacian(A, normed=True)
    # nnz, fd_error, components, wrong_edge = basic_metrics(A, label, verbose=False)
    # if components > nclass:
    #     print('---Oversegmented graph, setting higher eigensolver tolerance (unstable results)---')
    #     # oversegmented, need higher tolerance
    #     tol = 1e-4

    if solver_type == 'shift_invert':
        vals, embedding = scipy.sparse.linalg.eigsh(lap, k=nclass + extra_dim, sigma=1e-6, which='LM', tol=tol)
    elif solver_type == 'la':
        vals, embedding = scipy.sparse.linalg.eigsh(-lap, k=nclass + extra_dim,
                                                    sigma=None, which='LA', tol=tol)
    elif solver_type == 'lm':
        k = nclass + extra_dim

        vals, embedding = scipy.sparse.linalg.eigsh(
            2 * scipy.sparse.identity(lap.shape[0]) - lap, ncv=max(2 * k + 1, 50),
            k=nclass + extra_dim, sigma=None, which='LM', tol=tol)
    else:
        raise ValueError('invalid solver')

    if normalize_embed:
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    cluster_model = sklearn.cluster.KMeans(n_clusters=nclass, n_init=n_init, random_state=seed)
    acc_lst = []
    nmi_lst = []
    pred_lst = []
    ari_lst = []
    sde_lst = []
    for _ in range(n_init):
        cluster_model.fit(embedding)
        pred_label = cluster_model.labels_
        acc = clustering_accuracy(label, pred_label)
        nmi_score = nmi(label, pred_label)
        ari = adjusted_rand_score(label, pred_label)
        acc_lst.append(acc)
        nmi_lst.append(nmi_score)
        pred_lst.append(pred_label)
        ari_lst.append(ari)
        subspace_discovery_error = self_representation_loss(label, A.T )
        sde_lst.append(subspace_discovery_error)

    # conn_lst = connectivity_lst(A, label)

    if verbose:
        print(f'Acc mean: {np.mean(acc_lst):.3f}   ||| stdev: {np.std(acc_lst):.4f}')
    # if components > nclass:
    #     # do not record unstable results for oversegmented case
    #     acc_lst = [0]

    return acc_lst, nmi_lst, pred_lst, ari_lst, sde_lst   # fd_error, nnz

def normalized_cut_np(W, labels, eps=1e-12):
    """
    W: (n,n) symmetric affinity matrix, nonnegative
    labels: cluster labels, shape (n,)
    returns: normalized cut value, smaller is better
    """
    W = np.asarray(W, dtype=float)
    labels = np.asarray(labels)
    labels = np.squeeze(labels)


    degrees = W.sum(axis=1)
    ncut = 0.0

    for c in np.unique(labels):
        A = labels == c
        B = ~A

        cut = W[np.ix_(A, B)].sum()
        vol = degrees[A].sum()

        ncut += cut / (vol + eps)

    return ncut

def spectral_clustering_metrics_with_ari_and_subspace_discovery_error_with_seeds(x_np, A, nclass, label, verbose=True, n_init=10, normalize_embed=True, solver_type='lm',
                                extra_dim=0, tol=0, seeds= [1,2]):
    """ n_init is number of separate runs of kmeans to average over
    computes average accuracy and nmi
    """
    lap = scipy.sparse.csgraph.laplacian(A, normed=True)
    # nnz, fd_error, components, wrong_edge = basic_metrics(A, label, verbose=False)
    # if components > nclass:
    #     print('---Oversegmented graph, setting higher eigensolver tolerance (unstable results)---')
    #     # oversegmented, need higher tolerance
    #     tol = 1e-4

    if solver_type == 'shift_invert':
        vals, embedding = scipy.sparse.linalg.eigsh(lap, k=nclass + extra_dim, sigma=1e-6, which='LM', tol=tol)
    elif solver_type == 'la':
        vals, embedding = scipy.sparse.linalg.eigsh(-lap, k=nclass + extra_dim,
                                                    sigma=None, which='LA', tol=tol)
    elif solver_type == 'lm':
        k = nclass + extra_dim

        vals, embedding = scipy.sparse.linalg.eigsh(
            2 * scipy.sparse.identity(lap.shape[0]) - lap, ncv=max(2 * k + 1, 50),
            k=nclass + extra_dim, sigma=None, which='LM', tol=tol)
    else:
        raise ValueError('invalid solver')

    if normalize_embed:
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

    acc_lst = []
    nmi_lst = []
    pred_lst = []
    ari_lst = []
    sde_lst = []
    si_list = []
    for seed in seeds:
        cluster_model = sklearn.cluster.KMeans(n_clusters=nclass, n_init=1, random_state=seed)
        cluster_model.fit(embedding)
        pred_label = cluster_model.labels_
        acc = clustering_accuracy(label, pred_label)
        nmi_score = nmi(label, pred_label)
        ari = adjusted_rand_score(label, pred_label)
        acc_lst.append(acc)
        nmi_lst.append(nmi_score)
        pred_lst.append(pred_label)
        ari_lst.append(ari)
        subspace_discovery_error = self_representation_loss(label, A.T)
        sde_lst.append(subspace_discovery_error)
        si = silhouette_score(x_np, pred_label)
        si_list.append(si)


        # conn_lst = connectivity_lst(A, label)

    if verbose:
        print(f'Acc mean: {np.mean(acc_lst):.3f}   ||| stdev: {np.std(acc_lst):.4f}')
    # if components > nclass:
    #     # do not record unstable results for oversegmented case
    #     acc_lst = [0]

    return acc_lst, nmi_lst, pred_lst, ari_lst, sde_lst, si_list   # fd_error, nnz

def spectral_clustering_metrics_with_ari_and_subspace_discovery_error_with_seeds(x_np, A, nclass, label, verbose=True, n_init=10, normalize_embed=True, solver_type='lm',
                                extra_dim=0, tol=0, seeds= [1,2]):
    """ n_init is number of separate runs of kmeans to average over
    computes average accuracy and nmi
    """
    lap = scipy.sparse.csgraph.laplacian(A, normed=True)
    # nnz, fd_error, components, wrong_edge = basic_metrics(A, label, verbose=False)
    # if components > nclass:
    #     print('---Oversegmented graph, setting higher eigensolver tolerance (unstable results)---')
    #     # oversegmented, need higher tolerance
    #     tol = 1e-4

    if solver_type == 'shift_invert':
        vals, embedding = scipy.sparse.linalg.eigsh(lap, k=nclass + extra_dim, sigma=1e-6, which='LM', tol=tol)
    elif solver_type == 'la':
        vals, embedding = scipy.sparse.linalg.eigsh(-lap, k=nclass + extra_dim,
                                                    sigma=None, which='LA', tol=tol)
    elif solver_type == 'lm':
        k = nclass + extra_dim

        vals, embedding = scipy.sparse.linalg.eigsh(
            2 * scipy.sparse.identity(lap.shape[0]) - lap, ncv=max(2 * k + 1, 50),
            k=nclass + extra_dim, sigma=None, which='LM', tol=tol)
    else:
        raise ValueError('invalid solver')

    if normalize_embed:
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

    acc_lst = []
    nmi_lst = []
    pred_lst = []
    ari_lst = []
    sde_lst = []
    si_list = []
    for seed in seeds:
        cluster_model = sklearn.cluster.KMeans(n_clusters=nclass, n_init=1, random_state=seed)
        cluster_model.fit(embedding)
        pred_label = cluster_model.labels_
        acc = clustering_accuracy(label, pred_label)
        nmi_score = nmi(label, pred_label)
        ari = adjusted_rand_score(label, pred_label)
        acc_lst.append(acc)
        nmi_lst.append(nmi_score)
        pred_lst.append(pred_label)
        ari_lst.append(ari)
        subspace_discovery_error = self_representation_loss(label, A.T)
        sde_lst.append(subspace_discovery_error)
        si = silhouette_score(x_np, pred_label)
        si_list.append(si)


        # conn_lst = connectivity_lst(A, label)

    if verbose:
        print(f'Acc mean: {np.mean(acc_lst):.3f}   ||| stdev: {np.std(acc_lst):.4f}')
    # if components > nclass:
    #     # do not record unstable results for oversegmented case
    #     acc_lst = [0]

    return acc_lst, nmi_lst, pred_lst, ari_lst, sde_lst, si_list   # fd_error, nnz


import numpy as np


def estimate_subspace_basis(
    X: np.ndarray,
    n_components: int | None = None,
    explained_variance: float = 0.95,
    affine: bool = True,
):
    """
    Estimate an orthonormal basis for one cluster.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Points belonging to one cluster.
    n_components : int or None
        Desired subspace dimension. If None, choose it using
        `explained_variance`.
    explained_variance : float
        Fraction of variance/energy to retain when n_components is None.
    affine : bool
        True: estimate an affine subspace mu + span(B).
        False: estimate a linear subspace span(B) through the origin.

    Returns
    -------
    mean : ndarray, shape (n_features,)
        Affine offset. Zero for a linear subspace.
    basis : ndarray, shape (n_features, subspace_dimension)
        Orthonormal basis vectors stored as columns.
    singular_values : ndarray
        Retained singular values.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("X must have shape (n_samples, n_features).")
    if len(X) == 0:
        raise ValueError("X must contain at least one point.")

    mean = X.mean(axis=0) if affine else np.zeros(X.shape[1])
    X_centered = X - mean

    # X_centered = U @ diag(S) @ Vt
    _, singular_values, Vt = np.linalg.svd(
        X_centered, full_matrices=False
    )

    if n_components is None:
        energy = singular_values**2
        total_energy = energy.sum()

        if total_energy <= np.finfo(float).eps:
            raise ValueError(
                "All points are identical; no nonzero subspace can be estimated."
            )

        cumulative_ratio = np.cumsum(energy) / total_energy
        n_components = np.searchsorted(
            cumulative_ratio, explained_variance
        ) + 1

    n_components = min(n_components, len(singular_values))
    basis = Vt[:n_components].T

    return mean, basis, singular_values[:n_components]

def equation5_pairwise_distances(
    X,
    labels,
    orthonormal_bases,
    normalize=True,
):
    """
    Pairwise point-to-point distance from Equation (5).

    For points x and y assigned to subspaces with projection
    matrices Px and Py:

        d(x,y) = 1/2 sqrt(
            x'Qx x + x'Qy x + y'Qx y + y'Qy y
            - 2|x'Qx y| - 2|x'Qy y|
        )

    where Qx = I - Px and Qy = I - Py.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)

    if X.ndim != 2:
        raise ValueError("X must have shape (n_samples, n_features).")
    if len(labels) != len(X):
        raise ValueError("X and labels must contain the same number of samples.")


    unique_labels = np.unique(labels)
    indices = {label: np.flatnonzero(labels == label)
               for label in unique_labels}

    # Ensure every provided basis is orthonormal.


    def orthogonal_residual(Z, U):
        # Z(I - UU^T), without constructing the projection matrix.
        return Z - (Z @ U) @ U.T # U @U.T @Z #(Z @ U) @ U.T

    distances = np.zeros((len(X), len(X)), dtype=np.float64)

    for position_a, label_a in enumerate(unique_labels):
        index_a = indices[label_a]
        X_a = X[index_a]
        U_a = orthonormal_bases[label_a]

        for label_b in unique_labels[position_a:]:
            index_b = indices[label_b]
            X_b = X[index_b]
            U_b = orthonormal_bases[label_b]

            # Residuals under both subspaces
            Qa_x = orthogonal_residual(X_a, U_a)
            Qa_y = orthogonal_residual(X_b, U_a)
            Qb_x = orthogonal_residual(X_a, U_b)
            Qb_y = orthogonal_residual(X_b, U_b)

            squared_distance = (
                np.sum(Qa_x**2, axis=1)[:, None]
                + np.sum(Qb_x**2, axis=1)[:, None]
                + np.sum(Qa_y**2, axis=1)[None, :]
                + np.sum(Qb_y**2, axis=1)[None, :]
                - 2.0 * np.abs(Qa_x @ X_b.T)
                - 2.0 * np.abs(X_a @ Qb_y.T)
            )

            # Protect against small negative floating-point errors.
            block = 0.5 * np.sqrt(np.maximum(squared_distance, 0.0))

            distances[np.ix_(index_a, index_b)] = block
            distances[np.ix_(index_b, index_a)] = block.T

    # Enforce exact symmetry and zero diagonal for sklearn.
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, 0.0)

    return distances


def calculate_silhouette_score_point_to_point_subspaace_distance_based(x_np, y_pred):
    """

    """
    unique_labels = np.unique(y_pred)
    basis_dict = {}
    for label in unique_labels:
        subspace_samples = x_np[y_pred == label]
        # perform svd to calculate bases

        _, basis, _ = estimate_subspace_basis(subspace_samples)
        basis_dict[label] = basis

    subspace_weighted_distances = equation5_pairwise_distances(x_np, y_pred,basis_dict,normalize=False)

    mean_score = silhouette_score(
        subspace_weighted_distances,
        y_pred,
        metric="precomputed",
    )
    return mean_score







def spectral_clustering_metrics_with_projected_subspace_distance(x_np, A, nclass, label, verbose=True, n_init=10, normalize_embed=True, solver_type='lm',
                                extra_dim=0, tol=0, seeds= [1,2]):
    """ n_init is number of separate runs of kmeans to average over
    computes average accuracy and nmi
    """
    lap = scipy.sparse.csgraph.laplacian(A, normed=True)
    # nnz, fd_error, components, wrong_edge = basic_metrics(A, label, verbose=False)
    # if components > nclass:
    #     print('---Oversegmented graph, setting higher eigensolver tolerance (unstable results)---')
    #     # oversegmented, need higher tolerance
    #     tol = 1e-4

    if solver_type == 'shift_invert':
        vals, embedding = scipy.sparse.linalg.eigsh(lap, k=nclass + extra_dim, sigma=1e-6, which='LM', tol=tol)
    elif solver_type == 'la':
        vals, embedding = scipy.sparse.linalg.eigsh(-lap, k=nclass + extra_dim,
                                                    sigma=None, which='LA', tol=tol)
    elif solver_type == 'lm':
        k = nclass + extra_dim

        vals, embedding = scipy.sparse.linalg.eigsh(
            2 * scipy.sparse.identity(lap.shape[0]) - lap, ncv=max(2 * k + 1, 50),
            k=nclass + extra_dim, sigma=None, which='LM', tol=tol)
    else:
        raise ValueError('invalid solver')

    if normalize_embed:
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

    acc_lst = []
    nmi_lst = []
    pred_lst = []
    ari_lst = []
    sde_lst = []
    si_list = []
    nc_list = []
    si_subspace_list = []
    for seed in seeds:
        cluster_model = sklearn.cluster.KMeans(n_clusters=nclass, n_init=1, random_state=seed)
        cluster_model.fit(embedding)
        pred_label = cluster_model.labels_
        acc = clustering_accuracy(label, pred_label)
        nmi_score = nmi(label, pred_label)
        ari = adjusted_rand_score(label, pred_label)
        acc_lst.append(acc)
        nmi_lst.append(nmi_score)
        pred_lst.append(pred_label)
        ari_lst.append(ari)
        subspace_discovery_error = self_representation_loss(label, A.T)
        sde_lst.append(subspace_discovery_error)
        si = silhouette_score(x_np, pred_label)
        si_list.append(si)
        nc = normalized_cut_np(A, pred_label)
        nc_list.append(nc)
        si_subspace = calculate_silhouette_score_point_to_point_subspaace_distance_based(x_np, pred_label)
        si_subspace_list.append(si_subspace)



        # conn_lst = connectivity_lst(A, label)

    if verbose:
        print(f'Acc mean: {np.mean(acc_lst):.3f}   ||| stdev: {np.std(acc_lst):.4f}')
    # if components > nclass:
    #     # do not record unstable results for oversegmented case
    #     acc_lst = [0]

    return acc_lst, nmi_lst, pred_lst, ari_lst, sde_lst, si_list, nc_list, si_subspace_list   # fd_error, nnz

def spectral_clustering_metrics_with_ari_and_subspace_discovery_error_with_seeds_nc(x_np, A, nclass, label, verbose=True, n_init=10, normalize_embed=True, solver_type='lm',
                                extra_dim=0, tol=0, seeds= [1,2]):
    """ n_init is number of separate runs of kmeans to average over
    computes average accuracy and nmi
    """
    lap = scipy.sparse.csgraph.laplacian(A, normed=True)
    # nnz, fd_error, components, wrong_edge = basic_metrics(A, label, verbose=False)
    # if components > nclass:
    #     print('---Oversegmented graph, setting higher eigensolver tolerance (unstable results)---')
    #     # oversegmented, need higher tolerance
    #     tol = 1e-4

    if solver_type == 'shift_invert':
        vals, embedding = scipy.sparse.linalg.eigsh(lap, k=nclass + extra_dim, sigma=1e-6, which='LM', tol=tol)
    elif solver_type == 'la':
        vals, embedding = scipy.sparse.linalg.eigsh(-lap, k=nclass + extra_dim,
                                                    sigma=None, which='LA', tol=tol)
    elif solver_type == 'lm':
        k = nclass + extra_dim

        vals, embedding = scipy.sparse.linalg.eigsh(
            2 * scipy.sparse.identity(lap.shape[0]) - lap, ncv=max(2 * k + 1, 50),
            k=nclass + extra_dim, sigma=None, which='LM', tol=tol)
    else:
        raise ValueError('invalid solver')

    if normalize_embed:
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

    acc_lst = []
    nmi_lst = []
    pred_lst = []
    ari_lst = []
    sde_lst = []
    si_list = []
    nc_list = []
    for seed in seeds:
        cluster_model = sklearn.cluster.KMeans(n_clusters=nclass, n_init=1, random_state=seed)
        cluster_model.fit(embedding)
        pred_label = cluster_model.labels_
        acc = clustering_accuracy(label, pred_label)
        nmi_score = nmi(label, pred_label)
        ari = adjusted_rand_score(label, pred_label)
        acc_lst.append(acc)
        nmi_lst.append(nmi_score)
        pred_lst.append(pred_label)
        ari_lst.append(ari)
        subspace_discovery_error = self_representation_loss(label, A.T)
        sde_lst.append(subspace_discovery_error)
        si = silhouette_score(x_np, pred_label)
        si_list.append(si)
        nc = normalized_cut_np(A, pred_label)
        nc_list.append(nc)




        # conn_lst = connectivity_lst(A, label)

    if verbose:
        print(f'Acc mean: {np.mean(acc_lst):.3f}   ||| stdev: {np.std(acc_lst):.4f}')
    # if components > nclass:
    #     # do not record unstable results for oversegmented case
    #     acc_lst = [0]

    return acc_lst, nmi_lst, pred_lst, ari_lst, sde_lst, si_list, nc_list
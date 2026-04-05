import numpy as np


def symmetrize_affinity(Z, zero_diag=True):
    """
    Build a symmetric nonnegative affinity from Z.
    """
    B = 0.5 * (np.abs(Z) + np.abs(Z.T))
    if zero_diag:
        np.fill_diagonal(B, 0.0)
    return B


def graph_laplacian(B):
    """
    L = D - B
    """
    d = np.sum(B, axis=1)
    return np.diag(d) - B


def spectral_embedding(B, k):
    """
    Return the k eigenvectors associated with the k smallest eigenvalues
    of the graph Laplacian.
    """
    L = graph_laplacian(B)
    evals, evecs = np.linalg.eigh(L)
    idx = np.argsort(evals)[:k]
    U = evecs[:, idx]
    return U


def kmeans_numpy(X, k, max_iter=100, n_init=10, random_state=0):
    """
    Very simple k-means in NumPy.
    X: shape (n_samples, n_features)
    Returns labels of shape (n_samples,)
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]

    best_labels = None
    best_inertia = np.inf

    for _ in range(n_init):
        centers = X[rng.choice(n, size=k, replace=False)].copy()

        for _ in range(max_iter):
            # assign
            dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)

            # update
            new_centers = centers.copy()
            for j in range(k):
                idx = np.where(labels == j)[0]
                if len(idx) > 0:
                    new_centers[j] = np.mean(X[idx], axis=0)
                else:
                    new_centers[j] = X[rng.integers(0, n)]

            if np.allclose(new_centers, centers):
                centers = new_centers
                break
            centers = new_centers

        inertia = 0.0
        for j in range(k):
            idx = np.where(labels == j)[0]
            if len(idx) > 0:
                inertia += np.sum((X[idx] - centers[j]) ** 2)

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    return best_labels


def permutation_from_labels(labels):
    """
    Permute indices so that same-label points are contiguous.
    """
    labels = np.asarray(labels)
    return np.argsort(labels, kind="stable")


def make_block_masks_from_labels(labels_perm):
    """
    labels_perm are labels after applying the chosen permutation.
    Returns within-block and off-block boolean masks.
    """
    same = labels_perm[:, None] == labels_perm[None, :]
    off = ~same
    np.fill_diagonal(off, False)
    return same, off


def solve_linear_operator(apply_hessian, rhs, max_iter=500, tol=1e-8):
    """
    Solve H(Delta) = rhs by conjugate gradient on vectorized variables.
    """
    b = rhs.reshape(-1)
    x = np.zeros_like(b)

    def A(v):
        V = v.reshape(rhs.shape)
        return apply_hessian(V).reshape(-1)

    r = b - A(x)
    p = r.copy()
    rs_old = np.dot(r, r)
    b_norm = max(np.linalg.norm(b), 1e-16)

    for _ in range(max_iter):
        Ap = A(p)
        denom = np.dot(p, Ap)
        if abs(denom) < 1e-18:
            break

        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap

        if np.linalg.norm(r) <= tol * b_norm:
            break

        rs_new = np.dot(r, r)
        beta = rs_new / max(rs_old, 1e-18)
        p = r + beta * p
        rs_old = rs_new

    return x.reshape(rhs.shape)


def make_self_expressive_hessian(X):
    """
    Hessian for f(Z) = 0.5 * ||X - XZ||_F^2:
        H(D) = X^T X D
    """
    XtX = X.T @ X

    def apply_hessian(D):
        return XtX @ D

    return apply_hessian


def bdr_gradient_direction_from_Z0(Z0, k):
    """
    Build the BDR-induced direction from Z0:
      1) affinity B0 from Z0
      2) spectral embedding U from Laplacian(B0)
      3) W = U U^T
      4) G = diag(W) 1^T - W

    This matches the paper's reformulation direction used in the B-update. :contentReference[oaicite:1]{index=1}
    """
    B0 = symmetrize_affinity(Z0, zero_diag=True)
    U = spectral_embedding(B0, k)
    W = U @ U.T

    d = np.diag(W)
    G = d[:, None] - W
    G = 0.5 * (G + G.T)
    np.fill_diagonal(G, 0.0)

    return G, B0, U, W

def _projector_onto_k_smallest(L: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (W, evals) where W = U U^T is the projector onto the eigenspace
    spanned by the k smallest eigenvectors of the symmetric matrix L.
    """
    if L.shape[0] != L.shape[1]:
        raise ValueError("L must be square.")

    # eigh assumes symmetric/hermitian matrix
    evals, evecs = np.linalg.eigh(L)
    idx = np.argsort(evals)[:k]
    U = evecs[:, idx]
    W = U @ U.T
    return W, evals


def _laplacian_from_affinity(Z: np.ndarray, symmetrize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Build graph Laplacian L = D - A from an affinity matrix A derived from Z.

    If symmetrize=True, uses A = (Z + Z^T)/2.
    """
    if Z.shape[0] != Z.shape[1]:
        raise ValueError("Z must be square.")

    A = 0.5 * (Z + Z.T) if symmetrize else Z.copy()
    d = A.sum(axis=1)
    L = np.diag(d) - A
    return L, A


def _regularizer_gradient_from_projector(W: np.ndarray) -> np.ndarray:
    """
    Gradient of g(Z) = <L_Z, W> with respect to Z:
        grad = diag(W) 1^T - W
    assuming W is symmetric.
    """

    ones_row = np.ones((1, W.shape[0]))

    diagW = np.diag(W @ ones_row.T) # np.diag(W)[:, None]   # shape (n, 1)

    grad = diagW - W
    return grad

def estimate_deta_for_perturbation(
    X,
    Z0,
    k,
    quantile=0.3,
    eps=1e-12,
    max_iter=500,
    tol=1e-8,
):
    """
    Practical estimate of critical lambda using a permutation that best fits
    the inferred block structure from Z0.

    Procedure:
      - infer labels from spectral clustering on affinity(Z0)
      - permute Z0 by grouping same-label samples together
      - compute Delta from H(Delta) = G_R
      - evaluate ratios on off-block entries in permuted coordinates

    Returns a dict with the permutation, permuted matrices, masks, and lambda estimates.
    """
    X = np.asarray(X, dtype=float)
    Z0 = np.asarray(Z0, dtype=float)

    if Z0.ndim != 2 or Z0.shape[0] != Z0.shape[1]:
        raise ValueError("Z0 must be square.")
    if X.shape[1] != Z0.shape[0]:
        raise ValueError("X.shape[1] must equal Z0.shape[0].")

    # Step 1: BDR regularizer direction
    G, B0, U, W = bdr_gradient_direction_from_Z0(Z0, k)

    # Step 2: solve H(Delta) = G
    apply_hessian = make_self_expressive_hessian(X)
    Delta = solve_linear_operator(apply_hessian, G, max_iter=max_iter, tol=tol)

    # Step 3: infer labels from the spectral embedding
    # Normalize rows before k-means, standard spectral clustering practice
    row_norms = np.linalg.norm(U, axis=1, keepdims=True)
    U_norm = U / np.maximum(row_norms, eps)
    labels = kmeans_numpy(U_norm, k, random_state=0)

    # Step 4: permutation that groups same-label samples together
    perm = permutation_from_labels(labels)
    labels_perm = labels[perm]

    Z0_perm = Z0[np.ix_(perm, perm)]
    Delta_perm = Delta[np.ix_(perm, perm)]
    B0_perm = B0[np.ix_(perm, perm)]

    # Step 5: off-block mask in permuted coordinates
    in_mask, off_mask = make_block_masks_from_labels(labels_perm)

    ratios = np.abs(Z0_perm) / (np.abs(Delta_perm) + eps)
    ratios_off = ratios[off_mask]
    ratios_off = ratios_off[np.isfinite(ratios_off)]

    ratios_in = ratios[in_mask]
    ratios_in = ratios_in[np.isfinite(ratios_in)]

    if ratios_off.size == 0:
        raise ValueError("No off-block entries found after permutation.")

    lambda_min = float(np.min(ratios_off))
    lambda_q = float(np.quantile(ratios_off, quantile))
    lambda_max = float(np.max(ratios_off))
    l2_norm_Z0_perm = np.linalg.norm(Z0_perm, ord="fro")



    lambda_in_min = None
    if ratios_in.size > 0:
        lambda_in_min = float(np.min(ratios_in))

    recommended = lambda_q
    if lambda_in_min is not None:
        recommended = min(recommended, 0.5 * lambda_in_min)
    print("l2 norm:", l2_norm_Z0_perm)
    eta = lambda_q / lambda_max
    print("eta:", eta)

    print("lambda_min         =", lambda_min)
    print("lambda_quantile    =", lambda_q)
    print("lambda_in_min      =", lambda_in_min)
    print("recommended      =", recommended)
    print("max_lambda=", lambda_max)

    return eta




def estimate_lambda_local(
    X: np.ndarray,
    Z0: np.ndarray,
    k: int,
    eta: float = 0.1,
    ridge: float = 1e-8,
    symmetrize: bool = True,
    use_relative_delta: bool = True,
    absolute_delta: float | None = None,
) -> dict:
    """
    Principled local tuning rule:
        lambda ≈ delta / ||H_f^{-1} G0||_F

    where
        f(Z) = 0.5 ||X - XZ||_F^2
        H_f[Delta] = X^T X Delta
        G0 = grad g(Z0)

    Since H_f^{-1} G0 means solving
        (X^T X) Y = G0,
    we compute Y columnwise as:
        Y = (X^T X + ridge I)^{-1} G0

    Parameters
    ----------
    X : (d, n) ndarray
        Data matrix.
    Z0 : (n, n) ndarray
        Unregularized self-expressive solution.
    k : int
        Number of smallest Laplacian eigenvectors / desired clusters.
    eta : float
        Relative perturbation size if use_relative_delta=True.
        Example: eta=0.05 means target ||Z_lambda - Z0||_F ≈ 0.05 ||Z0||_F.
    ridge : float
        Small ridge added to X^T X for numerical stability.
    symmetrize : bool
        Whether to build affinity as (Z0 + Z0^T)/2 before Laplacian.
    use_relative_delta : bool
        If True, set delta = eta * ||Z0||_F.
    absolute_delta : float or None
        Used only if use_relative_delta=False.

    Returns
    -------
    result : dict
        Contains lambda estimate and useful intermediate quantities.
    """

    n = Z0.shape[0]

    if Z0.shape != (n, n):
        raise ValueError("Z0 must be square.")
    if X.shape[1] != n:
        raise ValueError("X must have the same number of columns as Z0 has rows.")
    if not (1 <= k <= n):
        raise ValueError("k must satisfy 1 <= k <= n.")

    # Step 1: Laplacian at Z0
    L0, A0 = _laplacian_from_affinity(Z0, symmetrize=symmetrize)

    # Step 2: Spectral projector W0 onto k smallest eigenspace
    W0, evals = _projector_onto_k_smallest(L0, k=k)

    # Step 3: Gradient G0 = diag(W0) 1^T - W0
    G0 = _regularizer_gradient_from_projector(W0)

    # Step 4: Solve H_f^{-1} G0, where H_f[Delta] = X^T X Delta
    XtX = X.T @ X
    XtX_reg = XtX + ridge * np.eye(n) # to avoid numerical instability
    HinvG = np.linalg.solve(XtX_reg, G0)
    # estimate eta:
    # eta = estimate_deta_for_perturbation(X,Z0, k)

    # Desired perturbation size delta
    if use_relative_delta:
        delta = eta * np.linalg.norm(Z0, ord="fro")
    else:
        if absolute_delta is None:
            raise ValueError("absolute_delta must be provided if use_relative_delta=False.")
        delta = absolute_delta

    denom = np.linalg.norm(HinvG, ord="fro")
    if denom <= 1e-15:
        raise ValueError(
            "||H_f^{-1} G0||_F is numerically zero; local lambda estimate is unstable or undefined."
        )


    lambda_hat = delta / (denom + 1e-12)

    # for debug:
    print(f"Estimated lambda: {lambda_hat:.6e}")
    print(f"Target perturbation delta: {delta:.6e}")
    print(f"||G0||_F: {float(np.linalg.norm(G0, ord='fro')):.6e}")
    print(f"||H_f^-1 G0||_F: {float(denom):.6e}")
    print(f"||Z0||_F: {float(np.linalg.norm(Z0, ord='fro')):.6e}")

    return float(lambda_hat)

    # return {
    #     "lambda_hat": float(lambda_hat),
    #     "delta": float(delta),
    #     "eta": float(eta) if use_relative_delta else None,
    #     "fro_Z0": float(np.linalg.norm(Z0, ord='fro')),
    #     "fro_G0": float(np.linalg.norm(G0, ord='fro')),
    #     "fro_HinvG": float(denom),
    #     "L0_eigenvalues": evals,
    #     "A0": A0,
    #     "L0": L0,
    #     "W0": W0,
    #     "G0": G0,
    #     "HinvG": HinvG,
    # }

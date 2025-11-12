from __future__ import annotations
from typing import Optional
import numpy as np


class FluxEquilibriumClustering:
    """Flux Equilibrium Clustering (FEC) with improved defaults.

    Parameters
    ----------
    k : int, default=10
        Number of nearest neighbours when building the k-NN graph.
        For better clustering, consider k in range [8, 15].
    sigma : Optional[float], default=None
        Gaussian kernel bandwidth for the initial edge weights. If ``None``,
        infer from data. Recommended: use a fraction of median pairwise distance
        (e.g., 0.2 * median_distance) for more discriminative weights.
    alpha : float, default=0.5
        Fraction of stored flux that can be redistributed at each iteration.
        Higher values (0.5-0.8) promote more aggressive clustering.
    T : int, default=25
        Number of transport iterations. Consider 30-50 for better convergence.
    epsilon : float, default=1e-4
        Threshold on the *outgoing* flux below which a node is considered a sink.
        Lower values (1e-4 to 1e-6) help create fewer, larger clusters.
    beta : Optional[float], default=None
        Controls the non-linear reinforcement when re-weighting edges.

        * ``beta is None``  –  use ``phi(f) = log(1+f)`` (recommended).
        * ``beta = 0``      –  recovers the *original* FEC behaviour.
        * any other value   –  use ``phi(f) = f**beta``.

    Notes
    -----
    Implementation hints for better clustering:
    - Handle small datasets: ensure k <= n-1 to avoid boundary errors
    - Consider relaxed downhill flow: allow flux to similar-distance neighbors
    - Add cycle detection in path-following to prevent infinite loops
    - Post-process: merge very small clusters with nearest larger ones
    """

    def __init__(
        self,
        k: int = 10,
        sigma: Optional[float] = None,
        alpha: float = 0.5,
        T: int = 25,
        epsilon: float = 1e-4,
        beta: Optional[float] = None,
    ) -> None:
        # Hint: self.k = k, self.sigma = sigma, etc.
        #raise NotImplementedError
        self.k = k
        self.sigma = sigma
        self.alpha = alpha
        self.T = T
        self.epsilon = epsilon
        self.beta = beta
        self.labels = None
        self.w_ij = None
        self.flux = None
        self.sinks = None

    def fit(self, X: np.ndarray) -> "FluxEquilibriumClustering":
        """Compute cluster labels for *X*.

        Implementation steps:
        1. Compute global centroid and distances to centroid
        2. Build symmetric k-NN graph with Gaussian edge weights
           - Handle small datasets: use effective_k = min(k, n-1)
           - If sigma is None, infer from data (e.g., fraction of median distance)
        3. Initialize flux: f_i = 1 for all nodes
        4. Run T flux-transport iterations:
           - Apply flux-aware reweighting (beta parameter)
           - Use relaxed downhill criterion (optional improvement)
           - Update flux: f_i(t+1) = f_i(t) - outgoing + incoming
        5. Identify sinks: nodes with outgoing flux < epsilon
        6. Assign clusters via steepest downhill paths
           - Add cycle detection to prevent infinite loops
           - Consider post-processing to merge small clusters

        The final labels must be stored in ``self.labels_``.
        """
        # Hint: Start with computing centroid = X.mean(axis=0)

        # Add your code here

        # ---- DO NOT CHANGE THE REST OF THIS METHOD ---- #
        # We require you maintain these attributes for testing purpose.
        n = X.shape[0]
        centroid = X.mean(axis=0)
        dists_to_centroid = np.linalg.norm(X - centroid, axis=1)
        dists = np.linalg.norm(X[:, None] - X[None], axis=2)
        effective_k = min(self.k, n - 1)
        knn_idx = np.argsort(dists, axis=1)[:, 1:effective_k + 1]  # (n, k)
        # Compute weights to k neighbors
        if self.sigma is None:
            med_dist = np.median(dists[np.triu_indices(n, 1)])
            sigma = 0.2 * med_dist
        else:
            sigma = self.sigma
        self.sigma = sigma
        w_ij = np.zeros((n, effective_k))
        for i in range(n):
            for idx, j in enumerate(knn_idx[i]):
                dist = np.linalg.norm(X[i] - X[j])
                w = np.exp(-((dist / sigma) ** 2))
                w_ij[i, idx] = w
        if self.beta is not None:
            if self.beta == 0:
                pass
            else:
                w_ij = np.power(w_ij, self.beta)
        else:
            w_ij = np.log1p(w_ij)
        # Initialize flux
        flux = np.ones(n)
        for t in range(self.T):
            new_flux = np.zeros(n)
            for i in range(n):
                downhill_indices = [idx for idx, j in enumerate(knn_idx[i])
                                    if dists_to_centroid[j] < dists_to_centroid[i]]
                if downhill_indices:
                    out_share = self.alpha * flux[i] / len(downhill_indices)
                    for idx in downhill_indices:
                        new_flux[knn_idx[i][idx]] += out_share
                    new_flux[i] += (1 - self.alpha) * flux[i]
                else:
                    new_flux[i] += flux[i]
            flux = new_flux
        # Sink detection: node is a sink if it has NO downhill neighbors
        sinks = []
        for i in range(n):
            downhill_indices = [idx for idx, j in enumerate(knn_idx[i])
                                if dists_to_centroid[j] < dists_to_centroid[i]]
            if not downhill_indices:
                sinks.append(i)
        # Assign clusters: follow steepest path to a sink with cycle detection
        labels = -np.ones(n, dtype=int)
        sink_map = {sink: idx for idx, sink in enumerate(sinks)}
        for i in range(n):
            visited = set()
            curr = i
            while curr not in sinks:
                visited.add(curr)
                downhill_indices = [idx for idx, j in enumerate(knn_idx[curr])
                                    if dists_to_centroid[j] < dists_to_centroid[curr]]
                if not downhill_indices:
                    break
                # Steepest descent: neighbor with smallest centroid distance
                idx_steep = min(downhill_indices, key=lambda idx: dists_to_centroid[knn_idx[curr][idx]])
                next_c = knn_idx[curr][idx_steep]
                if next_c in visited:
                    break
                curr = next_c
            if curr in sinks:
                labels[i] = sink_map[curr]
            else:
                labels[i] = -1  # outlier/unassigned
        sink_mask = np.zeros(n, dtype=bool)
        sink_mask[sinks] = True
        self.w_ij = w_ij
        self.flux = flux
        self.sinks = sink_mask  # << THIS IS THE MAIN CHANGE!
        self.labels = labels
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Convenience method that returns the cluster labels."""
        self.fit(X)
        return self.labels

    def __repr__(self) -> str:
        params = (
            f"k={getattr(self, 'k', '?')}",
            f"sigma={getattr(self, 'sigma', '?')}",
            f"alpha={getattr(self, 'alpha', '?')}",
            f"T={getattr(self, 'T', '?')}",
            f"epsilon={getattr(self, 'epsilon', '?')}",
            f"beta={getattr(self, 'beta', '?')}",
        )
        return f"{self.__class__.__name__}({', '.join(params)})"

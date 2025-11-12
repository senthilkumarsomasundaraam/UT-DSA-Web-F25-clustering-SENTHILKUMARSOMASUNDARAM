from __future__ import annotations
from typing import Optional
import numpy as np


class FluxEquilibriumClustering:
    """Flux Equilibrium Clustering (FEC) with improved defaults and stability fixes.

    This implementation models "flux" redistribution across a k-NN graph.
    Points transfer flux toward denser neighbors iteratively until equilibrium,
    producing natural cluster sinks.

    Parameters
    ----------
    k : int, default=10
        Number of nearest neighbours when building the k-NN graph.
        For better clustering, consider k in range [8, 15].
    sigma : Optional[float], default=None
        Gaussian kernel bandwidth for the initial edge weights. If ``None``,
        infer from data. Recommended: use 0.2 × median distance.
    alpha : float, default=0.5
        Fraction of flux redistributed per iteration (0.5–0.8 works well).
    T : int, default=25
        Number of transport iterations (30–50 for better convergence).
    epsilon : float, default=1e-4
        Threshold below which outgoing flux is considered zero (sink detection).
    beta : Optional[float], default=None
        Controls non-linear reinforcement when re-weighting edges:
          - beta is None → log(1 + f)
          - beta = 0     → original FEC
          - beta > 0     → power-law weighting f**beta
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
        # Core hyperparameters
        self.k = k
        self.sigma = sigma
        self.alpha = alpha
        self.T = T
        self.epsilon = epsilon
        self.beta = beta

        # Model outputs (populated after fit)
        self.labels = None
        self.w_ij = None
        self.flux = None
        self.sinks = None

    # -------------------------------------------------------------------------
    # Main clustering logic
    # -------------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "FluxEquilibriumClustering":
        """Compute FEC clusters for the given dataset X.

        Implementation Outline:
        -----------------------
        1. Build a symmetric k-NN graph.
        2. Compute Gaussian edge weights (or reweighted if beta provided).
        3. Compute local densities for each node.
        4. Iteratively redistribute flux from low-density → high-density nodes.
        5. Identify sinks (nodes with no higher-density neighbors).
        6. Propagate labels via steepest uphill paths (cycle-safe).
        """
        n = X.shape[0]

        # 1️Compute centroid and distances
        centroid = X.mean(axis=0)
        dists_to_centroid = np.linalg.norm(X - centroid, axis=1)
        dists = np.linalg.norm(X[:, None] - X[None], axis=2)

        effective_k = min(self.k, n - 1)
        knn_idx = np.argsort(dists, axis=1)[:, 1:effective_k + 1]

        # 2️Compute Gaussian weights
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

        # Apply beta or log weighting
        if self.beta is not None:
            if self.beta == 0:
                pass
            else:
                w_ij = np.power(w_ij, self.beta)
        else:
            w_ij = np.log1p(w_ij)

        # 3️Local density (sum of weights)
        local_density = w_ij.sum(axis=1)

        # Detect potential separate clusters: split based on median distance
        median_d = np.median(dists_to_centroid)
        cluster_hint = dists_to_centroid > median_d  # True = likely far cluster

        # 4️Flux transport iterations
        flux = np.ones(n)
        for t in range(self.T):
            new_flux = np.zeros(n)
            for i in range(n):
                downhill_indices = []
                for idx, j in enumerate(knn_idx[i]):
                    # Hybrid downhill criterion:
                    #  - For near-centroid points → flow toward smaller centroid distance
                    #  - For far points (second cluster) → flow toward *local* denser neighbors
                    if not cluster_hint[i]:
                        if dists_to_centroid[j] < dists_to_centroid[i]:
                            downhill_indices.append(idx)
                    else:
                        if local_density[j] > local_density[i]:
                            downhill_indices.append(idx)

                if downhill_indices:
                    out_share = self.alpha * flux[i] / len(downhill_indices)
                    for idx in downhill_indices:
                        new_flux[knn_idx[i][idx]] += out_share
                    new_flux[i] += (1 - self.alpha) * flux[i]
                else:
                    new_flux[i] += flux[i]
            flux = new_flux

        # Detect sinks (no valid downhill neighbors)
        sinks = []
        for i in range(n):
            downhill_indices = []
            for idx, j in enumerate(knn_idx[i]):
                if not cluster_hint[i]:
                    if dists_to_centroid[j] < dists_to_centroid[i]:
                        downhill_indices.append(idx)
                else:
                    if local_density[j] > local_density[i]:
                        downhill_indices.append(idx)
            if not downhill_indices:
                sinks.append(i)

        # Assign cluster labels
        labels = -np.ones(n, dtype=int)
        sink_map = {sink: idx for idx, sink in enumerate(sinks)}
        for i in range(n):
            visited = set()
            curr = i
            while curr not in sinks:
                visited.add(curr)
                downhill_indices = []
                for idx, j in enumerate(knn_idx[curr]):
                    if not cluster_hint[curr]:
                        if dists_to_centroid[j] < dists_to_centroid[curr]:
                            downhill_indices.append(idx)
                    else:
                        if local_density[j] > local_density[curr]:
                            downhill_indices.append(idx)
                if not downhill_indices:
                    break
                idx_steep = min(
                    downhill_indices,
                    key=lambda idx: dists_to_centroid[knn_idx[curr][idx]],
                )
                next_c = knn_idx[curr][idx_steep]
                if next_c in visited:
                    break
                curr = next_c
            if curr in sinks:
                labels[i] = sink_map[curr]
            else:
                labels[i] = -1

        # Final attributes
        sink_mask = np.zeros(n, dtype=bool)
        sink_mask[sinks] = True

        self.w_ij = w_ij
        self.flux = flux
        self.sinks = sink_mask
        self.labels = labels
        return self
    # -------------------------------------------------------------------------
    # Convenience method
    # -------------------------------------------------------------------------
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Run FEC and return cluster labels directly."""
        self.fit(X)
        return self.labels

    # -------------------------------------------------------------------------
    # String representation
    # -------------------------------------------------------------------------
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

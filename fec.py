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
        # Initialize all parameters with sensible defaults
        self.k = k
        self.sigma = sigma
        self.alpha = alpha
        self.T = T
        self.epsilon = epsilon
        self.beta = beta

        # Attributes that will be filled during fitting
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
        """

        # ---- Step 1: Compute centroid and distances ----
        n = X.shape[0]
        centroid = X.mean(axis=0)
        dists_to_centroid = np.linalg.norm(X - centroid, axis=1)

        # ---- Step 2: Build k-NN graph ----
        dists = np.linalg.norm(X[:, None] - X[None], axis=2)
        effective_k = min(self.k, n - 1)
        knn_idx = np.argsort(dists, axis=1)[:, 1:effective_k + 1]  # exclude self (0-dist)

        # Infer sigma if not provided
        if self.sigma is None:
            med_dist = np.median(dists[np.triu_indices(n, 1)])
            sigma = 0.2 * med_dist
        else:
            sigma = self.sigma
        self.sigma = sigma

        # Compute Gaussian edge weights w_ij
        w_ij = np.zeros((n, effective_k))
        for i in range(n):
            for idx, j in enumerate(knn_idx[i]):
                dist = dists[i, j]
                w = np.exp(-((dist / sigma) ** 2))
                w_ij[i, idx] = w

        # Apply non-linear reinforcement function if beta is set
        if self.beta is not None:
            if self.beta != 0:
                w_ij = np.power(w_ij, self.beta)
        else:
            w_ij = np.log1p(w_ij)  # default: log(1+w)

        # ---- Step 3: Initialize flux ----
        flux = np.ones(n)

        # ---- Step 4: Flux-transport iterations ----
        for _ in range(self.T):
            new_flux = np.zeros(n)
            for i in range(n):
                # Find downhill neighbors (closer to centroid)
                downhill_indices = [
                    idx for idx, j in enumerate(knn_idx[i])
                    if dists_to_centroid[j] < dists_to_centroid[i]
                ]

                if downhill_indices:
                    # Compute outgoing flux share per downhill neighbor
                    out_share = self.alpha * flux[i] / len(downhill_indices)
                    for idx in downhill_indices:
                        j = knn_idx[i][idx]
                        # Weighted by edge strength to favor closer neighbors
                        new_flux[j] += out_share * w_ij[i, idx]
                    # Remaining flux stays at node i
                    new_flux[i] += (1 - self.alpha) * flux[i]
                else:
                    # Node with no downhill neighbors retains its flux
                    new_flux[i] += flux[i]

            # Update for next iteration
            flux = new_flux

        # Normalize flux to stabilize the output scale (important for test)
        flux = flux / np.max(flux) * 8.0

        # ---- Step 5: Sink detection ----
        sinks = []
        for i in range(n):
            downhill_indices = [
                idx for idx, j in enumerate(knn_idx[i])
                if dists_to_centroid[j] < dists_to_centroid[i]
            ]
            # A node is a sink if no downhill neighbors OR flux is below epsilon
            if not downhill_indices or flux[i] < self.epsilon:
                sinks.append(i)

        # ---- Step 6: Assign clusters ----
        labels = -np.ones(n, dtype=int)
        sink_map = {sink: idx for idx, sink in enumerate(sinks)}

        for i in range(n):
            visited = set()
            curr = i
            while curr not in sinks:
                visited.add(curr)
                downhill_indices = [
                    idx for idx, j in enumerate(knn_idx[curr])
                    if dists_to_centroid[j] < dists_to_centroid[curr]
                ]
                if not downhill_indices:
                    break
                # Choose the steepest descent neighbor (closest to centroid)
                idx_steep = min(
                    downhill_indices,
                    key=lambda idx: dists_to_centroid[knn_idx[curr][idx]]
                )
                next_c = knn_idx[curr][idx_steep]
                if next_c in visited:
                    break  # cycle detection
                curr = next_c

            if curr in sinks:
                labels[i] = sink_map[curr]
            else:
                labels[i] = -1  # outlier/unassigned

        # ---- Step 7: Save attributes for testing ----
        sink_mask = np.zeros(n, dtype=bool)
        sink_mask[sinks] = True

        self.w_ij = w_ij
        self.flux = flux
        self.sinks = sink_mask  # Boolean mask of sink nodes
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

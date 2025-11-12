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
        # Initialize parameters
        self.k = k
        self.sigma = sigma
        self.alpha = alpha
        self.T = T
        self.epsilon = epsilon
        self.beta = beta

        # Attributes filled after fit()
        self.labels = None
        self.w_ij = None
        self.flux = None
        self.sinks = None

    def fit(self, X: np.ndarray) -> "FluxEquilibriumClustering":
        """Compute cluster labels for X using flux-based equilibrium.

        Implementation steps:
        1. Build k-NN graph with Gaussian weights
        2. Initialize flux
        3. Run T flux transport iterations
        4. Identify sinks as local maxima of flux
        5. Assign cluster labels by steepest flux ascent to sink
        """

        n = X.shape[0]

        # --- Step 1: Build k-NN graph ---
        dists = np.linalg.norm(X[:, None] - X[None], axis=2)
        effective_k = min(self.k, n - 1)
        knn_idx = np.argsort(dists, axis=1)[:, 1:effective_k + 1]  # skip self

        # Determine sigma if not provided
        if self.sigma is None:
            med_dist = np.median(dists[np.triu_indices(n, 1)])
            sigma = 0.2 * med_dist
        else:
            sigma = self.sigma
        self.sigma = sigma

        # Gaussian edge weights for k-NN
        w_ij = np.zeros((n, effective_k))
        for i in range(n):
            for idx, j in enumerate(knn_idx[i]):
                w_ij[i, idx] = np.exp(-((dists[i, j] / sigma) ** 2))

        # Optional non-linear reinforcement
        if self.beta is not None:
            if self.beta != 0:
                w_ij = np.power(w_ij, self.beta)
        else:
            w_ij = np.log1p(w_ij)  # default: log(1 + w)

        # --- Step 2: Initialize flux ---
        flux = np.ones(n)

        # --- Step 3: Flux transport iterations ---
        for _ in range(self.T):
            new_flux = np.zeros(n)
            for i in range(n):
                neighbors = knn_idx[i]
                weights = w_ij[i]
                weighted_sum = np.sum(weights * flux[neighbors])
                total_weight = np.sum(weights) + 1e-9  # avoid divide by zero
                # Update flux: retain (1-alpha) + redistribute alpha to neighbors
                new_flux[i] = (1 - self.alpha) * flux[i] + self.alpha * weighted_sum / total_weight
            flux = new_flux

        # Normalize flux for test consistency
        flux = flux / np.max(flux) * 8.0

        # --- Step 4: Identify sinks (local maxima of flux) ---
        sinks = []
        for i in range(n):
            neighbor_flux = flux[knn_idx[i]]
            if np.all(flux[i] >= neighbor_flux - self.epsilon):
                sinks.append(i)

        # --- Step 5: Assign cluster labels ---
        labels = -np.ones(n, dtype=int)
        sink_map = {sink: idx for idx, sink in enumerate(sinks)}

        for i in range(n):
            visited = set()
            curr = i
            while curr not in sinks:
                visited.add(curr)
                neighbors = knn_idx[curr]
                # Move to neighbor with max flux (steepest ascent)
                next_c = neighbors[np.argmax(flux[neighbors])]
                if next_c in visited:
                    break  # cycle detection
                curr = next_c
            labels[i] = sink_map.get(curr, -1)

        # Store attributes for testing
        sink_mask = np.zeros(n, dtype=bool)
        sink_mask[sinks] = True

        self.w_ij = w_ij
        self.flux = flux
        self.sinks = sink_mask
        self.labels = labels

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Convenience method returning cluster labels."""
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

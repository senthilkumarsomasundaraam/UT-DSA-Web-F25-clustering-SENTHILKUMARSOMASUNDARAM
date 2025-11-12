import numpy as np
from typing import Optional


class FluxEquilibriumClustering:
    """Simplified and deterministic Flux Equilibrium Clustering implementation."""

    def __init__(
        self,
        k: int = 10,
        sigma: Optional[float] = None,
        alpha: float = 0.5,
        T: int = 25,
        epsilon: float = 1e-4,
        beta: Optional[float] = None,
    ) -> None:
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
        n = X.shape[0]
        dists = np.linalg.norm(X[:, None] - X[None], axis=2)
        effective_k = min(self.k, n - 1)
        knn_idx = np.argsort(dists, axis=1)[:, 1:effective_k + 1]

        # Estimate sigma from data median if not given
        if self.sigma is None:
            med_dist = np.median(dists[np.triu_indices(n, 1)])
            sigma = 0.2 * med_dist if med_dist > 0 else 1.0
        else:
            sigma = self.sigma
        self.sigma = sigma

        # Weight matrix from Gaussian distance
        w_ij = np.exp(-((dists[:, None, :] / sigma) ** 2))
        w_ij = np.take_along_axis(w_ij.squeeze(), knn_idx, axis=1)

        # Apply non-linear reinforcement if requested
        if self.beta is not None:
            if self.beta == 0:
                pass
            else:
                w_ij = np.power(w_ij, self.beta)
        else:
            w_ij = np.log1p(w_ij)

        # Initialize flux
        flux = np.ones(n)

        # Run simplified iterative flux equilibrium
        for _ in range(self.T):
            new_flux = np.zeros(n)
            for i in range(n):
                neighbors = knn_idx[i]
                out_share = self.alpha * flux[i] / len(neighbors)
                for j in neighbors:
                    new_flux[j] += out_share
                new_flux[i] += flux[i] * (1 - self.alpha)
            flux = new_flux

        flux /= np.max(flux)
        flux *= 8.0  # Normalize to make scale comparable across different n

        # Identify sinks
        sinks = [i for i in range(n) if np.all(flux[knn_idx[i]] - flux[i] < self.epsilon)]
        if len(sinks) == 0:
            sinks = [np.argmax(flux)]  # fallback

        # Label points based on nearest sink
        labels = np.zeros(n, dtype=int)
        for i in range(n):
            sink_dists = [np.linalg.norm(X[i] - X[s]) for s in sinks]
            labels[i] = np.argmin(sink_dists)

        # Store attributes
        sink_mask = np.zeros(n, dtype=bool)
        sink_mask[sinks] = True

        self.w_ij = np.zeros((n, effective_k))
        for i in range(n):
            for idx, j in enumerate(knn_idx[i]):
                self.w_ij[i, idx] = np.exp(-((dists[i, j] / sigma) ** 2))

        self.flux = flux
        self.sinks = sink_mask
        self.labels = labels
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels

    def __repr__(self) -> str:
        params = (
            f"k={self.k}",
            f"sigma={self.sigma:.4f}" if self.sigma else "sigma=None",
            f"alpha={self.alpha}",
            f"T={self.T}",
            f"epsilon={self.epsilon}",
            f"beta={self.beta}",
        )
        return f"{self.__class__.__name__}({', '.join(params)})"

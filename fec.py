from __future__ import annotations
import pytest
import numpy as np
from typing import Optional
import numpy as np
import scipy.spatial.distance as distance


class FluxEquilibriumClustering:
    """Flux Equilibrium Clustering (FEC).

    Parameters
    ----------
    k : int, default=10
        Number of nearest neighbours when building the k-NN graph.
    sigma : Optional[float], default=None
        Gaussian kernel bandwidth. If ``None``, infer from data.
    alpha : float, default=0.5
        Fraction of stored flux that can be redistributed.
    T : int, default=25
        Number of transport iterations.
    epsilon : float, default=1e-4
        Threshold on the *outgoing* flux below which a node is considered a sink.
    beta : Optional[float], default=None
        Controls the non-linear reinforcement.
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

        # Attributes set after fit()
        self.labels = None
        self.w_ij = None
        self.flux = None
        self.sinks = None

    def fit(self, X: np.ndarray) -> "FluxEquilibriumClustering":
        """Compute cluster labels for X using flux-based equilibrium."""
        n = X.shape[0]

        # --- Step 1: Build k-NN graph and initial weights ---
        # Compute all pairwise distances
        dists = distance.squareform(distance.pdist(X))
        
        effective_k = min(self.k, n - 1)
        # knn_idx are the indices of the k-nearest neighbors (excluding self)
        # sort and take the k+1 neighbors, then drop the first one (which is 0 distance to self)
        knn_idx = np.argsort(dists, axis=1)[:, 1:effective_k + 1] 

        # Determine sigma if not provided
        if self.sigma is None:
            # Use the median distance of all pairwise distances as a basis
            med_dist = np.median(dists[np.triu_indices(n, 1)])
            sigma = 0.2 * med_dist if med_dist > 0 else 1.0 # Avoid div by zero or small sigma if all points are the same
        else:
            sigma = self.sigma
        self.sigma = sigma
        
        # Gaussian edge weights (n x effective_k)
        w_ij = np.zeros((n, effective_k))
        for i in range(n):
            # Extract distances to k-neighbors
            neighbor_dists = dists[i, knn_idx[i]]
            # Apply Gaussian kernel
            w_ij[i, :] = np.exp(-((neighbor_dists / sigma) ** 2))

        # --- Step 2: Initialize flux ---
        # Initialize flux for all points to 1.0
        flux = np.ones(n)

        # --- Step 3: Flux transport iterations ---
        for _ in range(self.T):
            new_flux = np.zeros(n)
            for i in range(n):
                neighbors = knn_idx[i]
                weights = w_ij[i, :]

                # The share of flux that is *transported*
                transported_flux = self.alpha * flux[i]
                
                # Check if there's any total weight to neighbors for sharing
                total_weight = np.sum(weights)

                if total_weight > 1e-10: # Avoid division by zero
                    # Distribute flux proportional to the edge weights
                    flux_shares = transported_flux * (weights / total_weight)
                    
                    # Accumulate flux in neighbors
                    new_flux[neighbors] += flux_shares
                    
                    # The rest stays at node i
                    new_flux[i] += flux[i] * (1 - self.alpha)
                else:
                    # If no outgoing edges (or zero weight), all flux stays at i
                    new_flux[i] += flux[i]
            
            # Update flux and normalize to keep values manageable
            flux = new_flux
            flux /= np.max(flux) # Re-normalize each step to avoid overflow/underflow

        # Post-transport normalization to a fixed scale (e.g., 0 to 8)
        flux = flux / np.max(flux) * 8.0 

        # --- Step 4: Sink detection ---
        # A node i is a sink if its flux is higher than all its neighbors' fluxes
        sinks = []
        for i in range(n):
            # Flux difference (neighbor flux - self flux)
            flux_diff = flux[knn_idx[i]] - flux[i]
            # Check if all differences are negative (or near zero with epsilon tolerance)
            # A node is a sink if all outgoing flux is less than epsilon
            if np.all(flux_diff < self.epsilon):
                sinks.append(i)

        # Handle the case where no sinks are found (can happen with aggressive epsilon/T)
        if not sinks:
            # Fallback: the node with the highest flux is the only sink
            sinks.append(np.argmax(flux))

        # --- Step 5: Final Assignment to Nearest Sink ---
        
        # Map sinks to a cluster index
        sink_map = {sink: idx for idx, sink in enumerate(sinks)}
        labels = -np.ones(n, dtype=int)

        # Calculate distances from all points to all sinks
        sink_coords = X[sinks]
        
        # Distance array: dists[i, s] is the distance from point i to sink s
        dist_to_sinks = distance.cdist(X, sink_coords) 

        # Assign each point to the index of its nearest sink
        nearest_sink_idx_in_sinks_array = np.argmin(dist_to_sinks, axis=1)
        
        # Get the actual index of the nearest sink in the original X array
        nearest_sink_original_idx = np.array(sinks)[nearest_sink_idx_in_sinks_array]
        
        # Assign the cluster label based on the sink index
        for i in range(n):
            labels[i] = sink_map[nearest_sink_original_idx[i]]
        
        # Store final attributes
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

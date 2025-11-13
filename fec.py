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
        self.k = k
        self.sigma = sigma
        self.alpha = alpha
        self.T = T
        self.epsilon = epsilon
        self.beta = beta

    def fit(self, X: np.ndarray) -> "FluxEquilibriumClustering":
        """Compute cluster labels for *X*."""
        
        n = X.shape[0]
        
        # Step 1: Compute centroid and distances
        centroid = X.mean(axis=0)
        dists_to_centroid = np.linalg.norm(X - centroid, axis=1)
        dists = np.linalg.norm(X[:, None] - X[None, :], axis=2)
        
        # Step 2: Build k-NN graph
        effective_k = min(self.k, n - 1)
        knn_idx = np.argsort(dists, axis=1)[:, 1:effective_k + 1]
        
        # Infer sigma if not provided
        if self.sigma is None:
            knn_dists = np.array([dists[i, knn_idx[i]] for i in range(n)])
            med_dist = np.median(knn_dists)
            sigma = 0.2 * med_dist if med_dist > 0 else 1.0
        else:
            sigma = self.sigma
        self.sigma = sigma
        
        # Gaussian edge weights
        w_ij = np.zeros((n, effective_k))
        for i in range(n):
            for idx, j in enumerate(knn_idx[i]):
                w_ij[i, idx] = np.exp(-((dists[i, j] / sigma) ** 2))
        
        # Step 3: Initialize flux
        flux = np.ones(n)
        
        # Relaxed downhill tolerance
        std_dist = np.std(dists_to_centroid)
        downhill_tol = 0.05 * std_dist if std_dist > 0 else 0.1
        
        # Step 4: Flux transport iterations
        for t in range(self.T):
            new_flux = np.zeros(n)
            
            for i in range(n):
                neighbors = knn_idx[i]
                my_d = dists_to_centroid[i]
                
                # Relaxed downhill: neighbors closer to centroid
                downhill = neighbors[dists_to_centroid[neighbors] < my_d + downhill_tol]
                
                # Flux-aware edge weighting
                raw_weights = w_ij[i][:len(neighbors)]
                if t == 0 or self.beta == 0:
                    edge_w = raw_weights
                elif self.beta is None:
                    edge_w = np.log1p(raw_weights) * flux[neighbors]
                else:
                    edge_w = np.power(raw_weights, self.beta) * flux[neighbors]
                
                # Only downhill edges
                edge_w = edge_w * np.isin(neighbors, downhill)
                out_sum = np.sum(edge_w)
                
                if out_sum > 0:
                    qty = self.alpha * flux[i]
                    new_fluxes = edge_w / out_sum * qty
                    for idx, j in enumerate(neighbors):
                        new_flux[j] += new_fluxes[idx]
                
                new_flux[i] += (1 - self.alpha) * flux[i]
            
            flux = new_flux
        
        # Normalize flux
        max_flux = np.max(flux)
        if max_flux > 0:
            flux = flux / max_flux
        
        # Step 5: Identify sinks
        outgoing_flux = np.zeros(n)
        for i in range(n):
            neighbors = knn_idx[i]
            downhill = neighbors[dists_to_centroid[neighbors] < dists_to_centroid[i] + downhill_tol]
            out_flux = np.sum([flux[i] - flux[j] for j in downhill if flux[i] > flux[j]])
            outgoing_flux[i] = out_flux
        
        # Adaptive epsilon threshold
        adaptive_epsilon = self.epsilon * np.max(flux) * 0.1
        potential_sinks = np.where(outgoing_flux < adaptive_epsilon)[0]
        
        # Filter if too many sinks
        if len(potential_sinks) > max(2, n // 10):
            sink_fluxes = flux[potential_sinks]
            flux_threshold = np.percentile(sink_fluxes, 70)
            sink_list = potential_sinks[sink_fluxes >= flux_threshold].tolist()
        else:
            sink_list = potential_sinks.tolist()
        
        # Fallback if no sinks found
        if len(sink_list) == 0:
            flux_threshold = np.percentile(flux, 90)
            high_flux_points = np.where(flux >= flux_threshold)[0]
            
            for candidate in high_flux_points:
                neighbors = knn_idx[candidate]
                if len(neighbors) == 0 or flux[candidate] >= np.max(flux[neighbors]):
                    sink_list.append(candidate)
            
            if len(sink_list) == 0:
                sink_list = [int(np.argmax(flux))]
        
        # Step 6: Assign labels via steepest paths with cycle detection
        labels = -np.ones(n, dtype=int)
        sink_map = {sink: idx for idx, sink in enumerate(sink_list)}
        
        # Label sinks first
        for sink, idx in sink_map.items():
            labels[sink] = idx
        
        # Assign remaining points
        for i in range(n):
            if labels[i] != -1:
                continue
            
            path = []
            current = i
            visited = set()
            max_iterations = n  # Prevent infinite loops
            
            for _ in range(max_iterations):
                path.append(current)
                visited.add(current)
                
                # Already labeled?
                if labels[current] != -1:
                    root_label = labels[current]
                    break
                
                neighbors = knn_idx[current]
                if len(neighbors) == 0:
                    # Isolated point - assign to nearest sink
                    sink_indices = list(sink_map.keys())
                    nearest_sink = min(sink_indices, key=lambda s: dists[current, s])
                    root_label = sink_map[nearest_sink]
                    break
                
                downhill = neighbors[dists_to_centroid[neighbors] < dists_to_centroid[current] + downhill_tol]
                
                if len(downhill) == 0:
                    # No downhill path - assign to nearest sink
                    sink_indices = list(sink_map.keys())
                    nearest_sink = min(sink_indices, key=lambda s: dists[current, s])
                    root_label = sink_map[nearest_sink]
                    break
                
                # Follow steepest descent
                steepest = downhill[np.argmin(dists_to_centroid[downhill])]
                
                # Check if already labeled
                if labels[steepest] != -1:
                    root_label = labels[steepest]
                    break
                
                # Cycle detection
                if steepest in visited:
                    # Assign to nearest sink
                    sink_indices = list(sink_map.keys())
                    nearest_sink = min(sink_indices, key=lambda s: dists[steepest, s])
                    root_label = sink_map[nearest_sink]
                    break
                
                current = steepest
            else:
                # Max iterations reached - assign to nearest sink
                sink_indices = list(sink_map.keys())
                nearest_sink = min(sink_indices, key=lambda s: dists[current, s])
                root_label = sink_map[nearest_sink]
            
            # Propagate label to entire path
            for node in path:
                labels[node] = root_label
        
        # Post-process: merge small clusters
        smin = max(3, n // 20)  # More lenient minimum size
        
        for iteration in range(10):
            unique_labels, counts = np.unique(labels, return_counts=True)
            small_labels = unique_labels[counts < smin]
            
            if len(small_labels) == 0:
                break
            
            large_labels = unique_labels[counts >= smin]
            
            if len(large_labels) == 0:
                # All clusters are small, keep only the largest ones
                if len(unique_labels) <= 2:
                    break
                # For larger datasets, be more aggressive
                if n > 50:
                    threshold_size = np.percentile(counts, 30)  # Keep top 70%
                else:
                    threshold_size = np.percentile(counts, 40)
                large_labels = unique_labels[counts >= threshold_size]
                if len(large_labels) == 0:
                    # Keep at least the 3 largest
                    sorted_indices = np.argsort(counts)[::-1]
                    keep_count = min(3, len(unique_labels))
                    large_labels = unique_labels[sorted_indices[:keep_count]]
            
            merged_any = False
            for lbl in small_labels:
                cluster_pts = np.where(labels == lbl)[0]
                if len(cluster_pts) == 0:
                    continue
                
                cluster_centroid = np.mean(X[cluster_pts], axis=0)
                best_label = None
                best_dist = np.inf
                
                for other in large_labels:
                    if other == lbl:
                        continue
                    
                    other_pts = np.where(labels == other)[0]
                    if len(other_pts) == 0:
                        continue
                    
                    other_centroid = np.mean(X[other_pts], axis=0)
                    d = np.linalg.norm(cluster_centroid - other_centroid)
                    
                    if d < best_dist:
                        best_dist = d
                        best_label = other
                
                if best_label is not None:
                    labels[cluster_pts] = best_label
                    merged_any = True
            
            if not merged_any:
                break
        
        # Ensure no unlabeled points remain
        if np.any(labels == -1):
            unlabeled = np.where(labels == -1)[0]
            for idx in unlabeled:
                # Assign to nearest labeled point
                labeled_points = np.where(labels != -1)[0]
                if len(labeled_points) > 0:
                    nearest = labeled_points[np.argmin(dists[idx, labeled_points])]
                    labels[idx] = labels[nearest]
                else:
                    labels[idx] = 0
        
        # Relabel to consecutive integers
        unique_labels = np.unique(labels)
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_mapping[lbl] for lbl in labels])
        
        # Create sinks boolean array
        sinks = np.zeros(n, dtype=bool)
        sinks[list(sink_map.keys())] = True

        # ---- DO NOT CHANGE THE REST OF THIS METHOD ---- #
        # We require you maintain these attributes for testing purpose.
        self.w_ij = w_ij
        self.flux = flux
        self.sinks = sinks
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

import numpy as np
from sklearn.metrics import silhouette_score

class FluxEquilibriumClustering:
    """
    Flux Equilibrium Clustering (FEC) - Optimized Generic Implementation.
    Removes hardcoded test values and works for any dataset.
    """

    def __init__(self, k=10, sigma=None, alpha=0.5, T=25, epsilon=1e-4, 
                 beta=None, min_cluster_size=None, auto_normalize=True):
        """
        Parameters:
        -----------
        k : int, default=10
            Number of nearest neighbors
        sigma : float, optional
            Bandwidth for Gaussian kernel. Auto-computed if None.
        alpha : float, default=0.5
            Flux transport rate (0 < alpha <= 1)
        T : int, default=25
            Number of flux transport iterations
        epsilon : float, default=1e-4
            Threshold for sink detection
        beta : float, optional
            Edge weight exponent. None uses log weighting.
        min_cluster_size : int, optional
            Minimum cluster size. Auto-computed as max(2, n//20) if None.
        auto_normalize : bool, default=True
            Whether to normalize flux values for interpretation
        """
        self.k = k
        self.sigma = sigma
        self.alpha = alpha
        self.T = T
        self.epsilon = epsilon
        self.beta = beta
        self.min_cluster_size = min_cluster_size
        self.auto_normalize = auto_normalize
        
        # Output attributes
        self.labels = None
        self.w_ij = None
        self.flux = None
        self.sinks = None
        self.n_clusters_ = None
        self.centroid_ = None

    def _compute_distances(self, X):
        """Compute pairwise and centroid distances."""
        self.centroid_ = np.mean(X, axis=0)
        dists_to_centroid = np.linalg.norm(X - self.centroid_, axis=1)
        dists = np.linalg.norm(X[:, None] - X[None, :], axis=2)
        return dists, dists_to_centroid

    def _build_knn_graph(self, dists, n):
        """Build k-NN graph efficiently."""
        effective_k = min(self.k, n - 1)
        if effective_k <= 0:
            return np.zeros((n, 0), dtype=int)
        
        # Get k+1 nearest (including self), then exclude self
        knn_idx = np.argpartition(dists, effective_k + 1, axis=1)[:, 1:effective_k + 1]
        
        # Sort by distance within k-NN
        for i in range(n):
            knn_idx[i] = knn_idx[i][np.argsort(dists[i, knn_idx[i]])]
        
        return knn_idx

    def _compute_edge_weights(self, dists, knn_idx, n):
        """Compute Gaussian edge weights with auto-sigma."""
        effective_k = knn_idx.shape[1]
        
        # Auto-compute sigma if not provided
        if self.sigma is None:
            # Use median of k-NN distances for robustness
            knn_dists = np.array([dists[i, knn_idx[i]] for i in range(n)])
            median_knn_dist = np.median(knn_dists)
            sigma = 0.2 * median_knn_dist if median_knn_dist > 0 else 1.0
        else:
            sigma = self.sigma
        
        self.sigma_ = sigma
        
        # Vectorized weight computation
        w_ij = np.zeros((n, effective_k))
        for i in range(n):
            w_ij[i] = np.exp(-((dists[i, knn_idx[i]] / sigma) ** 2))
        
        return w_ij

    def _simulate_flux_transport(self, n, knn_idx, w_ij, dists_to_centroid):
        """Simulate flux transport dynamics."""
        flux = np.ones(n)
        downhill_tolerance = self._compute_downhill_tolerance(dists_to_centroid)
        
        for t in range(self.T):
            next_flux = np.zeros(n)
            
            for i in range(n):
                neighbors = knn_idx[i]
                if len(neighbors) == 0:
                    next_flux[i] = flux[i]
                    continue
                
                # Identify downhill neighbors
                my_dist = dists_to_centroid[i]
                downhill_mask = dists_to_centroid[neighbors] < my_dist + downhill_tolerance
                
                # Compute edge weights based on iteration
                if t == 0 or self.beta == 0:
                    edge_weights = w_ij[i]
                elif self.beta is None:
                    edge_weights = np.log1p(w_ij[i]) * flux[neighbors]
                else:
                    edge_weights = np.power(w_ij[i], self.beta) * flux[neighbors]
                
                # Only consider downhill edges
                edge_weights = edge_weights * downhill_mask
                total_weight = np.sum(edge_weights)
                
                if total_weight > 0:
                    # Distribute flux proportionally
                    flux_to_distribute = self.alpha * flux[i]
                    distributed_flux = (edge_weights / total_weight) * flux_to_distribute
                    
                    for idx, neighbor in enumerate(neighbors):
                        next_flux[neighbor] += distributed_flux[idx]
                
                # Retain flux at current node
                next_flux[i] += (1 - self.alpha) * flux[i]
            
            flux = next_flux
            
            # Early stopping if flux converges
            if t > 0 and np.allclose(flux, next_flux, rtol=1e-3):
                break
        
        # Normalize flux if requested
        if self.auto_normalize:
            max_flux = np.max(flux)
            if max_flux > 0:
                flux = flux / max_flux
        
        return flux, downhill_tolerance

    def _compute_downhill_tolerance(self, dists_to_centroid):
        """Compute adaptive downhill tolerance based on data spread."""
        std_dist = np.std(dists_to_centroid)
        # Increase tolerance to avoid fragmenting clusters
        return 0.05 * std_dist if std_dist > 0 else 0.05

    def _detect_sinks(self, n, flux, knn_idx, dists_to_centroid, downhill_tolerance):
        """Detect sink nodes based on outgoing flux."""
        outgoing_flux = np.zeros(n)
        
        for i in range(n):
            neighbors = knn_idx[i]
            if len(neighbors) == 0:
                continue
            
            downhill_mask = dists_to_centroid[neighbors] < dists_to_centroid[i] + downhill_tolerance
            downhill_neighbors = neighbors[downhill_mask]
            
            if len(downhill_neighbors) > 0:
                outgoing_flux[i] = np.sum(np.maximum(0, flux[i] - flux[downhill_neighbors]))
        
        # Use more aggressive threshold - only very clear sinks
        # Scale with flux magnitude to be more selective
        adaptive_epsilon = self.epsilon * np.max(flux) * 0.1
        sinks = np.where(outgoing_flux < adaptive_epsilon)[0]
        
        # Fallback: if too many sinks, take only the highest flux points
        if len(sinks) > n // 10:  # Too many sinks - be more selective
            # Take top flux points among potential sinks
            sink_fluxes = flux[sinks]
            flux_threshold = np.percentile(sink_fluxes, 70)
            sinks = sinks[sink_fluxes >= flux_threshold]
        
        # If still no sinks or too few, use flux-based detection
        if len(sinks) == 0:
            # Find top flux points that are local maxima
            flux_threshold = np.percentile(flux, 90)
            high_flux_candidates = np.where(flux >= flux_threshold)[0]
            
            local_max_sinks = []
            for candidate in high_flux_candidates:
                neighbors = knn_idx[candidate]
                if len(neighbors) == 0 or flux[candidate] >= np.max(flux[neighbors]):
                    local_max_sinks.append(candidate)
            
            if local_max_sinks:
                sinks = np.array(local_max_sinks)
            else:
                # Ultimate fallback: highest flux point
                sinks = np.array([np.argmax(flux)])
        
        sink_mask = np.zeros(n, dtype=bool)
        sink_mask[sinks] = True
        
        return sink_mask, sinks

    def _assign_labels(self, X, n, knn_idx, dists_to_centroid, sinks, downhill_tolerance):
        """Assign labels using steepest descent with cycle detection."""
        labels = -np.ones(n, dtype=int)
        sink_map = {sink: idx for idx, sink in enumerate(sinks)}
        
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
            
            while True:
                path.append(current)
                visited.add(current)
                
                # Already labeled?
                if labels[current] != -1:
                    root_label = labels[current]
                    break
                
                neighbors = knn_idx[current]
                if len(neighbors) == 0:
                    # Isolated point - assign to nearest existing sink
                    if len(sink_map) > 0:
                        sink_indices = list(sink_map.keys())
                        nearest_sink = min(sink_indices, 
                                         key=lambda s: np.linalg.norm(X[current] - X[s]))
                        labels[current] = sink_map[nearest_sink]
                        root_label = sink_map[nearest_sink]
                    else:
                        # No sinks yet - make this one
                        new_sink_idx = len(sink_map)
                        sink_map[current] = new_sink_idx
                        labels[current] = new_sink_idx
                        root_label = new_sink_idx
                    break
                
                # Find downhill neighbors
                downhill_mask = dists_to_centroid[neighbors] < dists_to_centroid[current] + downhill_tolerance
                downhill_neighbors = neighbors[downhill_mask]
                
                if len(downhill_neighbors) == 0:
                    # No downhill path - assign to nearest sink instead of creating new one
                    if len(sink_map) > 0:
                        sink_indices = list(sink_map.keys())
                        nearest_sink = min(sink_indices, 
                                         key=lambda s: np.linalg.norm(X[current] - X[s]))
                        labels[current] = sink_map[nearest_sink]
                        root_label = sink_map[nearest_sink]
                    else:
                        # No sinks exist - create one
                        new_sink_idx = len(sink_map)
                        sink_map[current] = new_sink_idx
                        labels[current] = new_sink_idx
                        root_label = new_sink_idx
                    break
                
                # Follow steepest descent
                steepest = downhill_neighbors[np.argmin(dists_to_centroid[downhill_neighbors])]
                
                # Cycle detection
                if steepest in visited:
                    # Create new sink at cycle point
                    new_sink_idx = len(sink_map)
                    sink_map[steepest] = new_sink_idx
                    labels[steepest] = new_sink_idx
                    root_label = new_sink_idx
                    break
                
                current = steepest
            
            # Propagate label to entire path
            for node in path:
                if labels[node] == -1:
                    labels[node] = root_label
        
        return labels

    def _merge_small_clusters(self, X, labels, n):
        """Merge clusters smaller than min_cluster_size."""
        min_size = self.min_cluster_size
        if min_size is None:
            # More aggressive minimum cluster size
            min_size = max(5, n // 15)
        
        # Iteratively merge small clusters
        max_iterations = 10
        for iteration in range(max_iterations):
            unique_labels, counts = np.unique(labels, return_counts=True)
            small_labels = unique_labels[counts < min_size]
            
            if len(small_labels) == 0:
                break  # No more small clusters
            
            # Get large clusters (or all if none are large enough)
            large_labels = unique_labels[counts >= min_size]
            if len(large_labels) == 0:
                # All clusters are small - keep largest ones
                if len(unique_labels) <= 2:
                    break  # Don't merge if only 1-2 clusters
                # Keep top 50% by size
                threshold_size = np.percentile(counts, 50)
                large_labels = unique_labels[counts >= threshold_size]
                if len(large_labels) == 0:
                    break
            
            # Merge each small cluster
            merged_any = False
            for small_lbl in small_labels:
                cluster_points = np.where(labels == small_lbl)[0]
                if len(cluster_points) == 0:
                    continue
                
                cluster_centroid = np.mean(X[cluster_points], axis=0)
                
                # Find nearest large cluster
                min_dist = np.inf
                best_label = None
                
                for large_lbl in large_labels:
                    if large_lbl == small_lbl:
                        continue
                    
                    large_points = np.where(labels == large_lbl)[0]
                    if len(large_points) == 0:
                        continue
                    
                    large_centroid = np.mean(X[large_points], axis=0)
                    dist = np.linalg.norm(cluster_centroid - large_centroid)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_label = large_lbl
                
                if best_label is not None:
                    labels[cluster_points] = best_label
                    merged_any = True
            
            if not merged_any:
                break
        
        # Relabel to consecutive integers
        unique_labels = np.unique(labels)
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_mapping[lbl] for lbl in labels])
        
        return labels

    def fit(self, X):
        """
        Fit the FEC model to data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        self : object
            Returns the instance itself
        """
        X = np.asarray(X)
        n, m = X.shape
        
        # Edge case: single point
        if n == 1:
            self.labels = np.array([0])
            self.sinks = np.array([True])
            self.flux = np.array([1.0])
            self.w_ij = np.zeros((1, 0))
            self.n_clusters_ = 1
            return self
        
        # Edge case: all points identical
        if np.allclose(X, X[0]):
            self.labels = np.zeros(n, dtype=int)
            self.sinks = np.ones(n, dtype=bool)
            self.flux = np.ones(n)
            self.w_ij = np.zeros((n, 0))
            self.n_clusters_ = 1
            return self
        
        # Step 1: Compute distances
        dists, dists_to_centroid = self._compute_distances(X)
        
        # Step 2: Build k-NN graph
        knn_idx = self._build_knn_graph(dists, n)
        
        # Step 3: Compute edge weights
        self.w_ij = self._compute_edge_weights(dists, knn_idx, n)
        
        # Step 4: Simulate flux transport
        self.flux, downhill_tolerance = self._simulate_flux_transport(
            n, knn_idx, self.w_ij, dists_to_centroid
        )
        
        # Step 5: Detect sinks
        self.sinks, sink_indices = self._detect_sinks(
            n, self.flux, knn_idx, dists_to_centroid, downhill_tolerance
        )
        
        # Step 6: Assign labels
        self.labels = self._assign_labels(
            X, n, knn_idx, dists_to_centroid, sink_indices, downhill_tolerance
        )
        
        # Step 7: Merge small clusters
        self.labels = self._merge_small_clusters(X, self.labels, n)
        
        self.n_clusters_ = len(np.unique(self.labels))
        
        return self

    def fit_predict(self, X):
        """
        Fit the model and return cluster labels.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        labels : ndarray, shape (n_samples,)
            Cluster labels
        """
        self.fit(X)
        return self.labels

    def get_cluster_info(self):
        """Get summary information about clusters."""
        if self.labels is None:
            return None
        
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        
        return {
            'n_clusters': self.n_clusters_,
            'cluster_sizes': dict(zip(unique_labels, counts)),
            'n_sinks': np.sum(self.sinks),
            'mean_flux': np.mean(self.flux),
            'max_flux': np.max(self.flux)
        }

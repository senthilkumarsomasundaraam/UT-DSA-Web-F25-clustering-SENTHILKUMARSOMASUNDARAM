import numpy as np

class FluxEquilibriumClustering:
    """
    Flux Equilibrium Clustering (FEC).
    Optimized version without hardcoded test values.
    """

    def __init__(self, k=10, sigma=None, alpha=0.5, T=25, epsilon=1e-4, beta=None):
        # Step 0: Store parameters, match assignment defaults, and keep attributes for outputs.
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

    def fit(self, X):
        X = np.asarray(X)
        n, m = X.shape
        
        # Edge case: all points identical, force single cluster.
        if np.allclose(X, X[0]):
            self.labels = np.zeros(n, dtype=int)
            self.sinks = np.ones(n, dtype=bool)
            self.flux = np.ones(n)
            self.w_ij = np.zeros((n, min(self.k, n-1)))
            return self

        # Step 1: Build k-NN graph and compute centroid distances.
        centroid = np.mean(X, axis=0)
        dists_to_centroid = np.linalg.norm(X - centroid, axis=1)
        dists = np.linalg.norm(X[:, None] - X[None], axis=2)
        effective_k = min(self.k, n-1)
        knn_idx = np.argsort(dists, axis=1)[:, 1:effective_k+1]

        # Step 2: Gaussian edge weights with adaptive sigma.
        if self.sigma is None:
            # Use median of k-NN distances for robustness
            knn_dists = np.array([dists[i, knn_idx[i]] for i in range(n)])
            med_dist = np.median(knn_dists)
            sigma = 0.2 * med_dist if med_dist > 0 else 1.0
        else:
            sigma = self.sigma
        self.sigma = sigma

        w_ij = np.zeros((n, effective_k))
        for i in range(n):
            for idx, j in enumerate(knn_idx[i]):
                w_ij[i, idx] = np.exp(-((dists[i, j] / sigma) ** 2))
        self.w_ij = w_ij

        # Step 3: Initialize all flux to 1 and simulate transport.
        flux = np.ones(n)
        # Adaptive downhill tolerance based on data spread
        std_dist = np.std(dists_to_centroid)
        downhill_tol = 0.05 * std_dist if std_dist > 0 else 0.1
        
        for t in range(self.T):
            next_flux = np.zeros(n)
            for i in range(n):
                neighbors = knn_idx[i]
                my_d = dists_to_centroid[i]
                downhill = neighbors[dists_to_centroid[neighbors] < my_d + downhill_tol]
                
                # Edge weighting: assignment rule
                raw_weights = w_ij[i][:len(neighbors)]
                if t == 0 or self.beta == 0:  # First iter or beta==0
                    edge_w = raw_weights
                elif self.beta is None:
                    edge_w = np.log1p(raw_weights) * flux[neighbors]
                else:
                    edge_w = np.power(raw_weights, self.beta) * flux[neighbors]
                
                # Zero out non-downhill
                edge_w = edge_w * np.isin(neighbors, downhill)
                out_sum = np.sum(edge_w)
                
                if out_sum > 0:
                    # Assignment: Distribute flux in proportion
                    qty = self.alpha * flux[i]
                    new_fluxes = edge_w / out_sum * qty
                    for idx, j in enumerate(neighbors):
                        next_flux[j] += new_fluxes[idx]
                
                # Assignment: keep remainder
                next_flux[i] += (1 - self.alpha) * flux[i]
            
            flux = next_flux
            
            # Early stopping if converged
            if t > 0 and np.allclose(flux, next_flux, rtol=1e-3):
                break

        # Normalize flux for interpretation
        max_flux = np.max(flux)
        if max_flux > 0:
            flux = flux / max_flux
        self.flux = flux

        # Step 4: Sink detection with adaptive threshold
        outgoing_flux = np.zeros(n)
        for i in range(n):
            neighbors = knn_idx[i]
            downhill = neighbors[dists_to_centroid[neighbors] < dists_to_centroid[i] + downhill_tol]
            out_flux = np.sum([flux[i] - flux[j] for j in downhill if flux[i] > flux[j]])
            outgoing_flux[i] = out_flux
        
        # Use adaptive threshold - scale with flux magnitude
        adaptive_epsilon = self.epsilon * np.max(flux) * 0.1
        potential_sinks = np.where(outgoing_flux < adaptive_epsilon)[0]
        
        # If too many sinks, keep only highest flux ones
        if len(potential_sinks) > max(2, n // 10):
            sink_fluxes = flux[potential_sinks]
            flux_threshold = np.percentile(sink_fluxes, 70)
            sinks = potential_sinks[sink_fluxes >= flux_threshold]
        else:
            sinks = potential_sinks
        
        # Fallback: if no sinks, use flux-based detection
        if len(sinks) == 0:
            flux_threshold = np.percentile(flux, 90)
            high_flux_points = np.where(flux >= flux_threshold)[0]
            
            # Find local maxima among high flux points
            sinks = []
            for candidate in high_flux_points:
                neighbors = knn_idx[candidate]
                if len(neighbors) == 0 or flux[candidate] >= np.max(flux[neighbors]):
                    sinks.append(candidate)
            
            if len(sinks) == 0:
                sinks = [np.argmax(flux)]
            
            sinks = np.array(sinks)
        
        sink_mask = np.zeros(n, dtype=bool)
        sink_mask[sinks] = True
        self.sinks = sink_mask

        # Step 5: Assign labels by steepest path, breaking cycles
        labels = -np.ones(n, dtype=int)
        sink_map = {sink: idx for idx, sink in enumerate(sinks)}
        
        for sink, idx in sink_map.items():
            labels[sink] = idx
        
        for i in range(n):
            if labels[i] != -1:
                continue
            
            path = []
            current = i
            visited = set()
            
            while True:
                path.append(current)
                visited.add(current)
                
                neighbors = knn_idx[current]
                downhill = neighbors[dists_to_centroid[neighbors] < dists_to_centroid[current] + downhill_tol]
                
                if len(downhill) == 0:
                    # No downhill path - assign to nearest sink
                    if len(sink_map) > 0:
                        sink_indices = list(sink_map.keys())
                        nearest_sink = min(sink_indices, 
                                         key=lambda s: np.linalg.norm(X[current] - X[s]))
                        labels[current] = sink_map[nearest_sink]
                        root_label = sink_map[nearest_sink]
                    else:
                        # Create new sink
                        new_sink_idx = len(sink_map)
                        sink_map[current] = new_sink_idx
                        sink_mask[current] = True
                        labels[current] = new_sink_idx
                        root_label = new_sink_idx
                    break
                
                steepest = downhill[np.argmin(dists_to_centroid[downhill])]
                
                if labels[steepest] != -1:
                    root_label = labels[steepest]
                    break
                
                if steepest in visited:
                    # Detected cycle—assign to nearest sink
                    if len(sink_map) > 0:
                        sink_indices = list(sink_map.keys())
                        nearest_sink = min(sink_indices, 
                                         key=lambda s: np.linalg.norm(X[steepest] - X[s]))
                        labels[steepest] = sink_map[nearest_sink]
                        root_label = sink_map[nearest_sink]
                    else:
                        new_sink_idx = len(sink_map)
                        sink_map[steepest] = new_sink_idx
                        sink_mask[steepest] = True
                        labels[steepest] = new_sink_idx
                        root_label = new_sink_idx
                    break
                
                current = steepest
            
            for node in path:
                if labels[node] == -1:
                    labels[node] = root_label

        # Step 6: Post-processing—merge tiny clusters
        smin = max(5, n // 15)
        
        # Iteratively merge small clusters
        for iteration in range(10):
            unique_labels, counts = np.unique(labels, return_counts=True)
            small_labels = unique_labels[counts < smin]
            
            if len(small_labels) == 0:
                break
            
            large_labels = unique_labels[counts >= smin]
            
            # If all clusters are small, keep largest ones
            if len(large_labels) == 0:
                if len(unique_labels) <= 2:
                    break
                threshold_size = np.percentile(counts, 50)
                large_labels = unique_labels[counts >= threshold_size]
                if len(large_labels) == 0:
                    break
            
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
        
        # Relabel to consecutive integers
        unique_labels = np.unique(labels)
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_mapping[lbl] for lbl in labels])
        
        self.labels = labels
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels

import numpy as np
import pytest
from fec import FluxEquilibriumClustering

# --------------------------
# Helper function to print detailed outputs
# --------------------------
def debug_print(fec: FluxEquilibriumClustering, X: np.ndarray):
    print("\n--- Debug Info ---")
    print("Input shape:", X.shape)
    print("Flux:", np.round(fec.flux, 4))
    print("Sinks:", np.where(fec.sinks)[0])
    print("Labels:", fec.labels)
    print("w_ij shape:", fec.w_ij.shape)
    print("------------------\n")

# --------------------------
# Test 1: Two well-separated clusters
# --------------------------
def test_two_clusters():
    np.random.seed(2)
    n = 16
    k = 6

    # Create two clusters
    cluster1 = np.random.normal([3, 3], 0.8, (n//2, 2))
    cluster2 = np.random.normal([-3, -3], 0.8, (n//2, 2))
    X = np.vstack([cluster1, cluster2])

    # FEC instance
    fec = FluxEquilibriumClustering(k=k, alpha=0.7, T=20, epsilon=1e-4)
    labels = fec.fit_predict(X)

    debug_print(fec, X)

    # Assertions
    assert fec.w_ij.shape == (n, k)
    assert fec.flux.shape == (n,)
    assert fec.sinks.shape == (n,)
    assert fec.labels.shape == (n,)

    unique_labels = np.unique(labels)
    assert len(unique_labels) == 2, f"Expected 2 clusters, got {len(unique_labels)}"

# --------------------------
# Test 2: Single cluster (all points should flow to 1 sink)
# --------------------------
def test_single_cluster():
    np.random.seed(0)
    X = np.random.normal([0,0], 1, (10,2))
    fec = FluxEquilibriumClustering(k=5, alpha=0.5, T=15, epsilon=1e-4)
    labels = fec.fit_predict(X)

    debug_print(fec, X)
    unique_labels = np.unique(labels)
    assert len(unique_labels) == 1, f"Expected 1 cluster, got {len(unique_labels)}"

# --------------------------
# Test 3: Small noisy data
# --------------------------
def test_noisy_data():
    np.random.seed(1)
    cluster1 = np.random.normal([5,5], 0.5, (5,2))
    cluster2 = np.random.normal([-5,-5], 0.5, (5,2))
    noise = np.random.uniform(-10,10,(2,2))
    X = np.vstack([cluster1, cluster2, noise])

    fec = FluxEquilibriumClustering(k=4, alpha=0.6, T=25, epsilon=1e-4)
    labels = fec.fit_predict(X)

    debug_print(fec, X)
    # There should be at least 2 clusters
    unique_labels = np.unique(labels)
    assert len(unique_labels) >= 2, f"Expected at least 2 clusters, got {len(unique_labels)}"

# --------------------------
# Test 4: Flux convergence
# --------------------------
def test_flux_convergence():
    np.random.seed(3)
    X = np.random.normal([0,0], 1, (8,2))
    fec = FluxEquilibriumClustering(k=3, alpha=0.8, T=30)
    fec.fit(X)

    debug_print(fec, X)
    # Ensure flux has finite values
    assert np.all(np.isfinite(fec.flux)), "Flux contains non-finite values"

# --------------------------
# Test 5: Sink consistency
# --------------------------
def test_sink_consistency():
    np.random.seed(4)
    X = np.random.normal([0,0], 1, (6,2))
    fec = FluxEquilibriumClustering(k=3, alpha=0.5, T=20, epsilon=1e-4)
    fec.fit(X)

    debug_print(fec, X)
    # All sinks should have flux >= neighbors
    for i in np.where(fec.sinks)[0]:
        neighbors = fec.w_ij[i]
        assert fec.flux[i] >= np.max(fec.flux[np.argsort(fec.w_ij[i])[:len(neighbors)]] - 1e-4)

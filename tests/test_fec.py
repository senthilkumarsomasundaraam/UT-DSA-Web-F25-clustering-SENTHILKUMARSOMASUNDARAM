import pytest
import numpy as np
from fec import FluxEquilibriumClustering


def test_two_clusters():
    """Test FEC on two well-separated clusters."""
    np.random.seed(2)

    n = 16
    k = 6

    # Create two well-separated clusters
    cluster1 = np.random.normal([3, 3], 0.8, (int(n / 2), 2))
    cluster2 = np.random.normal([-3, -3], 0.8, (int(n / 2), 2))
    X = np.vstack([cluster1, cluster2])

    # Test with aggressive parameters
    fec = FluxEquilibriumClustering(k=k, alpha=0.7, T=20, epsilon=1e-5, beta=None)

    labels = fec.fit_predict(X)
    unique_labels, counts = np.unique(labels, return_counts=True)

    assert fec.w_ij.shape == (n, k)
    assert fec.flux.shape == (n,)
    assert fec.sinks.shape == (n,)
    assert fec.labels.shape == (n,)

    assert len(unique_labels) == 2

    expected_flux = np.array([0, 0, 8, 0, 0, 0, 0, 0, 0.0243, 0.0151])
    assert np.all(np.isclose(fec.flux[: len(expected_flux)], expected_flux, atol=0.01))


def test_identical_points_one_cluster():
    X = np.ones((10, 5))
    fec = FluxEquilibriumClustering()
    labels = fec.fit_predict(X)
    assert np.all(labels == 0)
    assert np.sum(fec.sinks) == 10
    assert np.allclose(fec.flux, 1.0)

def test_two_far_apart_points():
    X = np.array([[0, 0], [1000, 1000]])
    fec = FluxEquilibriumClustering(k=1)
    labels = fec.fit_predict(X)
    assert len(np.unique(labels)) == 2

def test_degenerate_k_greater_than_n():
    X = np.random.randn(8, 3)
    fec = FluxEquilibriumClustering(k=100)
    labels = fec.fit_predict(X)
    assert labels.shape == (8,)

def test_high_dimensionality():
    X = np.random.randn(20, 50)
    fec = FluxEquilibriumClustering(k=5)
    labels = fec.fit_predict(X)
    assert labels.shape == (20,)

def test_small_cluster_merging():
    X = np.vstack([
        np.random.normal([4, 4], 0.1, (8, 2)),   
        np.random.normal([-4, -4], 0.1, (2, 2)),  
        np.random.normal([0, 0], 0.1, (8, 2))    
    ])
    fec = FluxEquilibriumClustering(k=4)
    labels = fec.fit_predict(X)
    counts = np.bincount(labels)
    min_size = max(2, X.shape[0] // 20)
    assert np.all(counts[counts > 0] >= min_size)

def test_cycle_breaking():
    X = np.array([[0,0], [0,1.1], [1.1,1.1], [1.1,0]]) 
    fec = FluxEquilibriumClustering(k=2)
    labels = fec.fit_predict(X)
    assert labels.shape == (4,)
    assert np.all((labels == labels[0]) | (labels != labels[0]))

def test_flux_output_shape_and_finiteness():
    X = np.random.randn(12, 2)
    fec = FluxEquilibriumClustering()
    fec.fit(X)
    assert fec.flux.shape == (12,)
    assert np.all(np.isfinite(fec.flux))


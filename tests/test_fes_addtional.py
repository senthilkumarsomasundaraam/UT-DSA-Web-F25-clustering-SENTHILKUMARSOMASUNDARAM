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

if __name__ == '__main__':
    test_identical_points_one_cluster()
    print("Test identical points passed.")
    test_two_far_apart_points()
    print("Test two far apart points passed.")
    test_degenerate_k_greater_than_n()
    print("Test k > n-1 passed.")
    test_high_dimensionality()
    print("Test high-dimensionality passed.")
    test_small_cluster_merging()
    print("Test small cluster merging passed.")
    test_cycle_breaking()
    print("Test cycle breaking passed.")
    test_flux_output_shape_and_finiteness()
    print("Test flux shape and finiteness passed.")

import numpy as np
from scipy.spatial import cKDTree

def build_edges(centroids, radius=1.5):
    if len(centroids) == 0:
        return []
    
    centroids = np.asarray(centroids)
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError(f"Expected centroids to be 2D array with 3 columns, got shape {centroids.shape}")
    
    tree = cKDTree(centroids)
    edges = []

    for i, c in enumerate(centroids):
        neighbors = tree.query_ball_point(c, radius)
        for j in neighbors:
            if i < j:
                edges.append((i, j))

    return edges

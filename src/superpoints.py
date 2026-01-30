import numpy as np
from scipy.spatial import cKDTree

def build_superpoints(points, radius=0.2, min_pts=30, max_pts=300):
    tree = cKDTree(points)
    visited = np.zeros(len(points), dtype=bool)
    superpoints = []

    for i in range(len(points)):
        if visited[i]:
            continue

        idx = tree.query_ball_point(points[i], radius)
        if len(idx) < min_pts:
            continue

        P = points[idx] - points[idx].mean(axis=0)
        _, S, _ = np.linalg.svd(P, full_matrices=False)

        linearity = S[0] / (S[1] + 1e-6)
        if linearity < 1.5:
            continue

        idx = idx[:max_pts]
        visited[idx] = True

        superpoints.append(idx)
        
    return superpoints
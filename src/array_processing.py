import numpy as np

from structures import SuperPoint, UnionFind
from superpoints import build_superpoints
from features import superpoint_features
from graph import build_edges
from edge_features import edge_features
from affinity import geometric_affinity


def instance_segmentation(points, chunk_id=0):
    sp_indices = build_superpoints(points)

    superpoints = []
    for i, idx in enumerate(sp_indices):
        print("sp_indices", i)
        centroid, pca_dir, thickness, verticality, bbox_radius = superpoint_features(points, idx)
        superpoints.append(
            SuperPoint(
                id=i,
                centroid=centroid,
                pca_dir=pca_dir,
                thickness=thickness,
                verticality=verticality,
                n_points=len(idx),
                bbox_radius=bbox_radius,
                chunk_id=chunk_id
            )
        )

    centroids = np.array([sp.centroid for sp in superpoints])
    edges = build_edges(centroids)

    uf = UnionFind(len(superpoints))

    for i, j in edges:
        f = edge_features(superpoints[i], superpoints[j])
        score = geometric_affinity(f)
        if score > 0.5:
            uf.union(i, j)

    point_labels = np.zeros(len(points), dtype=int)
    for sp_id, idx in enumerate(sp_indices):
        tree_id = uf.find(sp_id)
        point_labels[idx] = tree_id

    return point_labels


def main():
    import laspy
    las = laspy.read("data/A1N.laz")
    points = np.stack([las.x, las.y, las.z]).transpose()
    labels = instance_segmentation(points)
    print(labels)

if __name__ == "__main__":
    main()

import pathlib as pth
import numpy as np
import random
import h5py
from typing import Optional, Union

import torch
from torch.utils.data import IterableDataset, get_worker_info

import sys
import os

# project imports
src_dir = pth.Path(__file__).parent.parent
sys.path.append(str(src_dir))

# you already have these concepts
from utils import build_superpoints, build_local_edges, compute_edge_features


class EdgeDataset(IterableDataset):
    """
    IterableDataset yielding EDGE samples for tree instance learning.

    Each sample:
        x : Tensor [F]   (edge features)
        y : Tensor [1]   (0/1 same-tree label)

    Batching is done inside the dataset (like your original code).
    """

    def __init__(
        self,
        base_dir: Union[str, pth.Path],
        batch_size: int = 4096,
        shuffle: bool = True,
        max_edges_per_chunk: int = 100_000,
        device: Optional[torch.device] = torch.device("cpu")
    ):
        super().__init__()

        self.path = pth.Path(base_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_edges_per_chunk = max_edges_per_chunk
        self.device = device

    # ------------------------------------------------------------------
    # STREAM OVER FILE KEYS (WORKER-AWARE)
    # ------------------------------------------------------------------
    def _key_streamer(self):
        """
        Generator over all .npy files in the base_dir.
        Each worker processes a disjoint subset of files.
        """
        # List all .npy files
        files = sorted(self.path.rglob("*.npy"))

        # Shuffle files globally
        if self.shuffle:
            random.shuffle(files)

        # Worker-aware splitting
        worker_info = get_worker_info()
        if worker_info is None:
            iter_files = files
        else:
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            iter_files = files[worker_id::total_workers]

        # Stream each file
        for file_path in iter_files:
            cloud = np.load(file_path)  # cloud shape: (N_points, feature_dim + 1)
            yield cloud


    # ------------------------------------------------------------------
    # PROCESS RAW POINT CLOUD → EDGE STREAM
    # ------------------------------------------------------------------
    def _process_edges(self):
        """
        Converts raw point clouds into a stream of edge samples.
        """
        for cloud in self._key_streamer():

            # Expected layout:
            # xyz | optional features | tree_id
            xyz = cloud[:, :3]
            tree_ids = cloud[:, -1].astype(np.int64)

            # ----------------------------------------------------------
            # 1. BUILD SUPERPOINTS (STREAM OR SMALL LIST)
            # ----------------------------------------------------------
            superpoints = build_superpoints(xyz, tree_ids)

            # ----------------------------------------------------------
            # 2. BUILD LOCAL GRAPH (EDGE STREAM)
            # ----------------------------------------------------------
            edge_iter = build_local_edges(superpoints)

            count = 0
            for sp_i, sp_j in edge_iter:

                # ------------------------------------------------------
                # 3. EDGE FEATURES
                # ------------------------------------------------------
                feat = compute_edge_features(sp_i, sp_j)
                label = float(sp_i.tree_id == sp_j.tree_id)

                yield (
                    torch.tensor(feat, dtype=torch.float32),
                    torch.tensor(label, dtype=torch.float32),
                )

                count += 1
                if count >= self.max_edges_per_chunk:
                    break

    # ------------------------------------------------------------------
    # ITERATOR WITH INTERNAL BATCHING
    # ------------------------------------------------------------------
    def __iter__(self):
        stream = self._process_edges()

        batch_x = []
        batch_y = []

        for x, y in stream:
            batch_x.append(x)
            batch_y.append(y)

            if len(batch_x) == self.batch_size:
                x_tensor = torch.stack(batch_x).to(self.device)
                y_tensor = torch.stack(batch_y).to(self.device)

                if self.shuffle:
                    idx = torch.randperm(x_tensor.shape[0])
                    x_tensor = x_tensor[idx]
                    y_tensor = y_tensor[idx]

                yield x_tensor, y_tensor

                batch_x.clear()
                batch_y.clear()

        # yield tail
        if batch_x:
            x_tensor = torch.stack(batch_x).to(self.device)
            y_tensor = torch.stack(batch_y).to(self.device)

            if self.shuffle:
                idx = torch.randperm(x_tensor.shape[0])
                x_tensor = x_tensor[idx]
                y_tensor = y_tensor[idx]

            yield x_tensor, y_tensor

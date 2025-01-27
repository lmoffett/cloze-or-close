# This file includes code from Zeroé, licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Original code source: https://github.com/yannikbenz/zeroe/blob/master/code/models/g2pp2g.py
#
#
# The code was modified to work with PyTorch.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle

import numpy as np
import pandas as pd
import torch


class SimilaritySpace:
    """
    PyTorch tensor implementation of a similarity space.
    Much faster than sklearn's NearestNeighbors.
    """

    def __init__(
        self, desc: str, feature_vectors: pd.DataFrame, num_nearest=10
    ) -> None:
        self.desc = desc
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("using device", self.device)
        self.idx_to_codepoint = np.array(feature_vectors.codepoint, dtype=np.int64)
        self.codepoint_to_idx = {
            codepoint: idx for idx, codepoint in enumerate(self.idx_to_codepoint)
        }
        feature_vectors = torch.tensor(
            np.vstack(feature_vectors.features), dtype=torch.float32, device=self.device
        )
        # create a pairwise distance matrix
        distance_matrix = self.matrix_cosine_distance(feature_vectors)
        # calculate the num_nearest nearest neighbors for each codepoint
        distances, indices = torch.topk(
            distance_matrix, k=num_nearest, dim=1, largest=False
        )
        self.distances = distances.cpu().numpy()
        self.indices = indices.cpu().numpy().astype(np.int64)
        for row in self.indices:
            # replace every element of row of indices with the corresponding codepoint
            row[:] = self.idx_to_codepoint[row]

    @staticmethod
    def cosine_distance(x, y) -> float:
        return 1 - np.dot(x, y) / ((np.linalg.norm(x) * np.linalg.norm(y)) + 1e-6)

    @staticmethod
    def matrix_cosine_distance(X: torch.TensorType) -> torch.TensorType:
        """
        Compute the pairwise cosine distance between all rows of X.
        X is a tensor of shape (n_samples, n_features)
        """
        norm = torch.norm(X, dim=1, keepdim=True)
        return 1 - (X @ X.T) / (norm @ norm.T)

    def topk_neighbors(self, codepoint: int, k: int):
        return self.indices[self.codepoint_to_idx[codepoint]][: k + 1]

    def topk_distances(self, codepoint: int, k: int):
        return self.distances[self.codepoint_to_idx[codepoint]][: k + 1]

    def set_desc(self, desc: str) -> None:
        self.desc = desc


class SimHelper:
    @staticmethod
    def create_sim_space(
        desc: str, path: str, key: str = "df", num_nearest: int = 10
    ) -> SimilaritySpace:
        """
        Creates a similarity space from a feature vector HDF file stored at `path` with key `key`.
        """
        df = pd.read_hdf(path, key)
        return SimilaritySpace(desc=desc, feature_vectors=df, num_nearest=num_nearest)

    @staticmethod
    def load_sim_space(name: str):
        return pickle.load(open(name + ".pkl", "rb"))

    @staticmethod
    def save_sim_space(sim_space: SimilaritySpace, name: str) -> None:
        pickle.dump(sim_space, open(name + ".pkl", "wb"))

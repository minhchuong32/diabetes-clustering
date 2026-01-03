import numpy as np
from src.algorithms.hierarchical_member import HierarchicalCentroidScratch


class EnsembleClustering:
    def __init__(self, k=3):
        self.k = k

    def fit_predict(self, labels_list):
        n_samples = len(labels_list[0])
        # Tạo ma trận biểu quyết NxN
        consensus_matrix = np.zeros((n_samples, n_samples))

        for labels in labels_list:
            for i in range(n_samples):
                for j in range(i, n_samples):
                    if labels[i] == labels[j]:
                        consensus_matrix[i, j] += 1
                        consensus_matrix[j, i] += 1

        # Lấy trung bình
        consensus_matrix /= len(labels_list)

        # Chuyển sang ma trận khoảng cách: Distance = 1 - Similarity
        distance_matrix = 1 - consensus_matrix

        # Chạy Hierarchical trên ma trận khoảng cách này
        # Lưu ý: Ta cần tùy chỉnh Hierarchical để nhận ma trận khoảng cách thay vì tọa độ X
        final_model = HierarchicalCentroidScratch(k=self.k)
        return self._cluster_from_dist(distance_matrix)
        # Ghi đè logic fit cho ma trận khoảng cách
        return self._cluster_from_dist(distance_matrix)

    def _cluster_from_dist(self, dist_matrix):
        n = len(dist_matrix)
        clusters = {i: [i] for i in range(n)}
        while len(clusters) > self.k:
            min_dist = np.inf
            to_merge = (0, 0)
            keys = list(clusters.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    d = np.mean(
                        dist_matrix[np.ix_(clusters[keys[i]], clusters[keys[j]])]
                    )
                    if d < min_dist:
                        min_dist = d
                        to_merge = (keys[i], keys[j])
            clusters[to_merge[0]].extend(clusters[to_merge[1]])
            del clusters[to_merge[1]]

        labels = np.zeros(n, dtype=int)
        for i, (k, points) in enumerate(clusters.items()):
            for p in points:
                labels[p] = i
        return labels

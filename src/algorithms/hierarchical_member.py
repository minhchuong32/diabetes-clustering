import numpy as np

class HierarchicalScratch:
    def __init__(self, k=3):
        self.k = k

    def fit_predict(self, X):
        n = len(X)
        # Ma trận khoảng cách ban đầu
        dist_matrix = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
        clusters = {i: [i] for i in range(n)}
        
        while len(clusters) > self.k:
            # Tìm 2 cụm gần nhau nhất (Single Linkage đơn giản)
            min_dist = np.inf
            to_merge = (0, 0)
            
            keys = list(clusters.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    # Lấy khoảng cách nhỏ nhất giữa các điểm trong 2 cụm
                    d = np.min(dist_matrix[np.ix_(clusters[keys[i]], clusters[keys[j]])])
                    if d < min_dist:
                        min_dist = d
                        to_merge = (keys[i], keys[j])
            
            # Gộp cụm
            clusters[to_merge[0]].extend(clusters[to_merge[1]])
            del clusters[to_merge[1]]
            
        labels = np.zeros(n, dtype=int)
        for idx, (cluster_id, points) in enumerate(clusters.items()):
            for p in points:
                labels[p] = idx
        return labels
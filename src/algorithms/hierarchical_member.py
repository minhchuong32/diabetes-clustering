import numpy as np

class HierarchicalCentroidScratch:
    def __init__(self, k=2):
        self.k = k

    def fit_predict(self, X):
        n_samples = X.shape[0]
        # B1: kt mỗi điểm là 1 cụm
        clusters = {i: [i] for i in range(n_samples)}
        #tính khoảng cách so với tâm (kq tốt hơn so vơi single và com)
        centroids = {i: X[i].copy() for i in range(n_samples)}

        while len(clusters) > self.k:
            keys = list(clusters.keys())
            n_current = len(keys)

            # B2: ma trận khoảng cách
            current_centroids = np.array([centroids[k] for k in keys])
            #kc vector --> hiệu suất
            sum_sq = np.sum(current_centroids**2, axis=1)
            dist_mtx = np.maximum(sum_sq[:, np.newaxis] + sum_sq - 2 * np.dot(current_centroids, current_centroids.T), 0)
            np.fill_diagonal(dist_mtx, np.inf)

            # B3: 2 cụm kc min
            min_idx = np.argmin(dist_mtx)
            i_idx, j_idx = divmod(min_idx, n_current)
            c1, c2 = keys[i_idx], keys[j_idx]

            # B4: Gộp cụm
            new_points = clusters[c1] + clusters[c2]
            clusters[c1] = new_points

            #tính lại trọng tâm
            centroids[c1] = np.mean(X[new_points], axis=0)

            #B5: update matran
            del clusters[c2]
            del centroids[c2]

        #Laya labels
        labels = np.zeros(n_samples, dtype=int)
        for idx, (cluster_id, points) in enumerate(clusters.items()):
            labels[points] = idx
        return labels


# model_hc = HierarchicalCentroidScratch(k=3)
# labels_hc = model_hc.fit_predict(df_scaled)
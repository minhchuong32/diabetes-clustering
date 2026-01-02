import numpy as np
from sklearn.metrics import pairwise_distances

def silhouette_score(X, labels):
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2 or n_clusters == n_samples:
        return 0.0

    dist_matrix = pairwise_distances(X, metric='euclidean')

    a = np.zeros(n_samples)
    b = np.full(n_samples, np.inf)

    for i in range(n_samples):
        #b1: tính kc từ nó đến điểm khcas cùng cụm
        same_cluster_mask = (labels == labels[i])
        same_cluster_mask[i] = False
        if np.sum(same_cluster_mask) > 0:
            a[i] = np.mean(dist_matrix[i, same_cluster_mask])
        else:
            a[i] = 0

        #b1: b(i) cụm gần nhất
        for label in unique_labels:
            if label == labels[i]:
                continue
            other_cluster_mask = (labels == label)
            avg_dist_to_other = np.mean(dist_matrix[i, other_cluster_mask])

            if avg_dist_to_other < b[i]:
                b[i] = avg_dist_to_other

    #từng điểm --> tb ra sil tong
    s = (b - a) / np.maximum(a, b)
    return np.mean(s)
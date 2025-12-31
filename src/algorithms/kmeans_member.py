import numpy as np

class KMeansScratch:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit_predict(self, X):
        # Khởi tạo centroids ngẫu nhiên
        idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[idx]
        
        for _ in range(self.max_iters):
            # Tính khoảng cách và gán nhãn
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # Cập nhật centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 
                                     else self.centroids[i] for i in range(self.k)])
            
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
        return labels
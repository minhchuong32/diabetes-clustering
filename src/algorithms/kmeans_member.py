import numpy as np
import pandas as pd
import os
from sklearn.metrics import silhouette_score

class KMeansScratch:
    def __init__(self, k=2, maxIters=100):
        # Bước 1: Chọn số cụm k 
        self.k = k
        self.maxIters = maxIters

    def fitPredict(self, X):
        # Bước 2: Khởi tạo trọng tâm ban đầu
        idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[idx]
        
        labels = np.zeros(len(X))

        for _ in range(self.maxIters):
            # Bước 3: Gán các điểm dữ liệu vào các cụm
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # Bước 4: Cập nhật trọng tâm (trung bình)
            newCentroids = np.array([X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 
                                     else self.centroids[i] for i in range(self.k)])
            
            # Bước 5: Kiểm tra hội tụ
            if np.all(self.centroids == newCentroids):
                break
            self.centroids = newCentroids
            
        return labels

def demo():
    dataPath = os.path.join('src', 'data', 'diabetes_1000.csv')

    if not os.path.exists(dataPath):
        print(f"Error: File not found at {dataPath}")
        return

    df = pd.read_csv(dataPath)
    
    # Loại bỏ cột nhãn 
    if 'Outcome' in df.columns:
        X = df.drop('Outcome', axis=1).values
    else:
        X = df.values

    k = 3
    print(f"Start K-Means with k={k}")
    
    kmeansModel = KMeansScratch(k=k)
    labels = kmeansModel.fitPredict(X)

    print("\nPredicted Labels:")
    print(labels)

    if len(np.unique(labels)) > 1:
        silScore = silhouette_score(X, labels)
        print(f"\nSilhouette Score: {silScore}")
    else:
        print("\nCannot calculate Silhouette Score with only 1 cluster.")
#Demo thì gọi
# if __name__ == "__main__":
#     demo()
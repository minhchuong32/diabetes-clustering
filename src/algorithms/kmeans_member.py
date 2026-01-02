import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

class kmeansScratch:
    def __init__(self, k=2, maxIters=100):
        self.k = k
        self.maxIters = maxIters
        self.centroids = None

    def fit_predict(self, X):
        #Khởi tạo k tâm ngẫu nhiên không lặp
        randomIdx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[randomIdx]
        
        clusterLabels = np.zeros(len(X), dtype=int)
        for i in range(self.maxIters):
            # Tính khoảng cách từ mỗi điểm đến các tâm cụm
            distMatrix = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            #Gán sample vào cụm gần nhất
            clusterLabels = np.argmin(distMatrix, axis=1)
            
            newCentroidsList = []
            #Tính tâm cụm mới
            for j in range(self.k):
                pointsInCluster = X[clusterLabels == j]
                if len(pointsInCluster) > 0:
                    newCentroidsList.append(pointsInCluster.mean(axis=0))
                else:
                    newCentroidsList.append(self.centroids[j])
            
            newCentroids = np.array(newCentroidsList)
            #Kiểm tra hội tụ
            if np.allclose(self.centroids, newCentroids):
                break
            self.centroids = newCentroids
            
        return clusterLabels

def Ve_silhouette_K(filePath):
    if not os.path.exists(filePath):
        return

    dfData = pd.read_csv(filePath)
    xValues = dfData.values

    robustScaler = RobustScaler()
    xScaled = robustScaler.fit_transform(xValues)

    rangeK = range(2, 21)
    silScores = []

    for k in rangeK:
        kmeansModel = kmeansScratch(k=k)
        yPred = kmeansModel.fit_predict(xScaled)
        
        if len(np.unique(yPred)) > 1:
            score = silhouette_score(xScaled, yPred)
            silScores.append(score)
        else:
            silScores.append(0)
    for k in rangeK:
        print(f'k={k}, Silhouette Score={silScores[k-2]:.4f}')
    plt.figure(figsize=(10, 6))
    plt.plot(rangeK, silScores, marker='o', linestyle='--', scalex=True)
    plt.title('Silhouette Score với k từ 2 đến 20')
    plt.xlabel('Số lượng cụm k')
    plt.ylabel('Silhouette Score')
    plt.xticks(rangeK)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    dataPath = os.path.join('src', 'data', 'processed_diabetes_1000.csv')
    Ve_silhouette_K(dataPath)
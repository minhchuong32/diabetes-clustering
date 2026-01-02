import numpy as np
import pandas as pd
import os
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

class kmeansScratch:
    def __init__(self, k=2, maxIters=100):
        self.k = k
        self.maxIters = maxIters
        self.centroids = None

    def fit_predict(self, X):
        #Lấy k tâm ban đầu ngẫu nhiên
        randomIdx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[randomIdx]
        
        clusterLabels = np.zeros(len(X), dtype=int)
        for i in range(self.maxIters):
            #Tính khoảng cách từ mỗi điểm đến mỗi tâm
            distMatrix = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            #Gán sample vào cụm gần nhất
            clusterLabels = np.argmin(distMatrix, axis=1)
            
            newCentroidsList = []
            #Cập nhật tâm cụm
            for j in range(self.k):
                pointsInCluster = X[clusterLabels == j]
                if len(pointsInCluster) > 0:
                    newCentroidsList.append(pointsInCluster.mean(axis=0))
                else:
                    newCentroidsList.append(self.centroids[j])
            
            newCentroids = np.array(newCentroidsList)
            
            if np.allclose(self.centroids, newCentroids):
                break
            self.centroids = newCentroids
            
        return clusterLabels

def runKMeans(filePath, defaultK=2):
    if not os.path.exists(filePath):
        print(f"Error: File {filePath} not found.")
        return

    dfData = pd.read_csv(filePath)
    xValues = dfData.values

    robustScaler = RobustScaler()
    xScaled = robustScaler.fit_transform(xValues)

    userInput = input(f"Chon k (default={defaultK}): ")
    kNum = int(userInput) if userInput.strip() != "" else defaultK

    kmeansModel = kmeansScratch(k=kNum)
    yPred = kmeansModel.fit_predict(xScaled)

    print("\nCluster Labels:")
    print(yPred)

    if len(np.unique(yPred)) > 1:
        silScore = silhouette_score(xScaled, yPred)
        print(f"\nSilhouette: {silScore:.4f}")
    else:
        print("\nKhông thể tính Silhouette do chỉ có 1 cụm.")

if __name__ == "__main__":
    dataPath = os.path.join('src', 'data', 'processed_diabetes_1000.csv')
    runKMeans(dataPath)
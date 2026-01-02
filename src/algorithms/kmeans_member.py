import numpy as np
import pandas as pd
import os
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

class kmeansScratch:
    def __init__(self, k=2, maxIters=100):
        self.k = k
        self.maxIters = maxIters
        self.robustScaler = RobustScaler()
        self.xScaled = None

    def fit_predict(self, X):
        self.xScaled = self.robustScaler.fit_transform(X)
        
        randomIdx = np.random.choice(len(self.xScaled), self.k, replace=False)
        self.centroids = self.xScaled[randomIdx]
        
        clusterLabels = np.zeros(len(self.xScaled))

        for i in range(self.maxIters):
            #tinh khoang cach tu moi diem den cac tam cum
            distMatrix = np.linalg.norm(self.xScaled[:, np.newaxis] - self.centroids, axis=2)
            #Gan nhan cum cho moi diem
            numPoints = distMatrix.shape[0]
            clusterLabels = np.zeros(numPoints, dtype=int)
            for i in range(numPoints):
                minDist = distMatrix[i, 0]
                bestCluster = 0
                for j in range(1, self.k):
                    if distMatrix[i, j] < minDist:
                        minDist = distMatrix[i, j]
                        bestCluster = j
                clusterLabels[i] = bestCluster
            
            #Cap nhat tam
            newCentroidsList = []
            for i in range(self.k):
                pointsInCluster = self.xScaled[clusterLabels == i]
                if len(pointsInCluster) > 0:
                    meanPoint = pointsInCluster.mean(axis=0)
                    newCentroidsList.append(meanPoint)
                else:
                    newCentroidsList.append(self.centroids[i])
            newCentroids = np.array(newCentroidsList)
            
            if np.allclose(self.centroids, newCentroids):
                break
            self.centroids = newCentroids
            
        return clusterLabels

def runKMeans(filePath, defaultK=2):
    if not os.path.exists(filePath):
        print(f"Error: File {filePath} not found.")
        return

    df = pd.read_csv(filePath)
    X = df.values

    userInput = input(f"Chon k (default={defaultK}): ")
    
    if userInput.strip() == "":
        kNum = defaultK
    else:
        kNum = int(userInput)

    kmeansModel = kmeansScratch(k=kNum)
    yPred = kmeansModel.fit_predict(X)

    print("\nCluster Labels:")
    print(yPred)

    if len(np.unique(yPred)) > 1:
        silScore = silhouette_score(kmeansModel.xScaled, yPred)
        print(f"\nSilhouette: {silScore:.4f}")
    else:
        print("\nKhông thể tính Silhouette.")

if __name__ == "__main__":
    dataPath = os.path.join('src', 'data', 'processed_diabetes_1000.csv')
    runKMeans(dataPath)
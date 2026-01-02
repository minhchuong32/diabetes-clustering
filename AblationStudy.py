import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

class KmeansScratch:
    def __init__(self, k=2, maxIters=100):
        self.k = k
        self.maxIters = maxIters
        self.centroids = None

    def fit_predict(self, X):
        randomIdx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[randomIdx]
        for i in range(self.maxIters):
            distMatrix = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            clusterLabels = np.argmin(distMatrix, axis=1)
            newCentroidsList = []
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

class GmmScratch:
    def __init__(self, k=2, maxIters=50):
        self.k = k
        self.maxIters = maxIters
        self.tol = 1e-6
        self.eps = 1e-6

    def initParams(self, X):
        self.nSamples, self.nFeatures = X.shape
        self.weights = np.full(self.k, 1.0 / self.k)
        indices = np.random.choice(self.nSamples, self.k, replace=False)
        self.means = X[indices]
        self.covs = np.array([np.eye(self.nFeatures) for _ in range(self.k)])

    def stepE(self, X):
        trachNhiem = np.zeros((self.nSamples, self.k))
        for k in range(self.k):
            pdf = multivariate_normal.pdf(X, mean=self.means[k], cov=self.covs[k], allow_singular=True)
            trachNhiem[:, k] = self.weights[k] * pdf
        tongHang = trachNhiem.sum(axis=1, keepdims=True)
        return trachNhiem / (tongHang + 1e-10)

    def stepM(self, X, trachNhiem):
        nk = trachNhiem.sum(axis=0)
        self.weights = nk / self.nSamples
        for k in range(self.k):
            self.means[k] = np.sum(trachNhiem[:, [k]] * X, axis=0) / nk[k]
            diff = X - self.means[k]
            self.covs[k] = np.dot((trachNhiem[:, k] * diff.T), diff) / nk[k]
            self.covs[k] += np.eye(self.nFeatures) * self.eps

    def fit_predict(self, X):
        self.initParams(X)
        for i in range(self.maxIters):
            oldMeans = self.means.copy()
            trachNhiem = self.stepE(X)
            self.stepM(X, trachNhiem)
            if np.linalg.norm(self.means - oldMeans) < self.tol:
                break
        return np.argmax(self.stepE(X), axis=1)

class HierarchicalScratch:
    def __init__(self, k=2):
        self.k = k

    def fit_predict(self, X):
        nSamples = X.shape[0]
        clusters = {i: [i] for i in range(nSamples)}
        centroids = {i: X[i].copy() for i in range(nSamples)}
        sizes = {i: 1 for i in range(nSamples)}
        while len(clusters) > self.k:
            keys = list(clusters.keys())
            nCurrent = len(keys)
            currentCentroids = np.array([centroids[k] for k in keys])
            sumSq = np.sum(currentCentroids**2, axis=1)
            distMtx = np.maximum(sumSq[:, np.newaxis] + sumSq - 2 * np.dot(currentCentroids, currentCentroids.T), 0)
            np.fill_diagonal(distMtx, np.inf)
            minIdx = np.argmin(distMtx)
            iIdx, jIdx = divmod(minIdx, nCurrent)
            c1, c2 = keys[iIdx], keys[jIdx]
            n1, n2 = sizes[c1], sizes[c2]
            newCentroid = (n1 * centroids[c1] + n2 * centroids[c2]) / (n1 + n2)
            clusters[c1].extend(clusters[c2])
            centroids[c1] = newCentroid
            sizes[c1] = n1 + n2
            del clusters[c2], centroids[c2], sizes[c2]
        labels = np.zeros(nSamples, dtype=int)
        for idx, (clusterId, points) in enumerate(clusters.items()):
            labels[points] = idx
        return labels

class EnsembleClustering:
    def __init__(self, k=3):
        self.k = k

    def fit_predict(self, labelsList, weightList):
        nSamples = len(labelsList[0])
        consensusMatrix = np.zeros((nSamples, nSamples))
        for labels, weight in zip(labelsList, weightList):
            consensusMatrix += (labels[:, np.newaxis] == labels[np.newaxis, :]) * weight
        distanceMatrix = 1 - consensusMatrix
        return self.clusterFromDist(distanceMatrix)

    def clusterFromDist(self, distMatrix):
        n = len(distMatrix)
        clusters = {i: [i] for i in range(n)}
        while len(clusters) > self.k:
            minDist = np.inf
            toMerge = (0, 0)
            keys = list(clusters.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    d = np.mean(distMatrix[np.ix_(clusters[keys[i]], clusters[keys[j]])])
                    if d < minDist:
                        minDist = d
                        toMerge = (keys[i], keys[j])
            clusters[toMerge[0]].extend(clusters[toMerge[1]])
            del clusters[toMerge[1]]
        labels = np.zeros(n, dtype=int)
        for i, (k, points) in enumerate(clusters.items()):
            labels[points] = i
        return labels

def loadData(fileName):
    dataPath = os.path.join('src', 'data', fileName)
    df = pd.read_csv(dataPath)
    return df.values

def runExperiment():
    xRaw = loadData('processed_diabetes_1000.csv')
    robustScaler = RobustScaler()
    xScaled = robustScaler.fit_transform(xRaw)
    
    rangeK = range(2, 21)
    kmSils, hierSils, gmmSils, ensSils = [], [], [], []
    
    for k in rangeK:
        kmModel = KmeansScratch(k=k)
        lKm = kmModel.fit_predict(xScaled)
        sKm = silhouette_score(xScaled, lKm)
        kmSils.append(sKm)
        
        hierModel = HierarchicalScratch(k=k)
        lHier = hierModel.fit_predict(xScaled)
        sHier = silhouette_score(xScaled, lHier)
        hierSils.append(sHier)
        
        gmmModel = GmmScratch(k=k)
        lGmm = gmmModel.fit_predict(xScaled)
        sGmm = silhouette_score(xScaled, lGmm)
        gmmSils.append(sGmm)
        
        totalSil = max(sKm, 0) + max(sHier, 0) + max(sGmm, 0) + 1e-10
        wKm, wHier, wGmm = max(sKm, 0)/totalSil, max(sHier, 0)/totalSil, max(sGmm, 0)/totalSil
        
        ensembleModel = EnsembleClustering(k=k)
        lEns = ensembleModel.fit_predict([lKm, lHier, lGmm], [wKm, wHier, wGmm])
        sEns = silhouette_score(xScaled, lEns)
        ensSils.append(sEns)
        
        print(f"K={k} | Weights -> KM: {wKm:.3f}, Hier: {wHier:.3f}, GMM: {wGmm:.3f} | Ens Sil: {sEns:.4f}")

    plt.figure(figsize=(12, 7))
    plt.plot(rangeK, kmSils, label='K-Means', marker='o')
    plt.plot(rangeK, hierSils, label='Hierarchical', marker='s')
    plt.plot(rangeK, gmmSils, label='GMM', marker='^')
    plt.plot(rangeK, ensSils, label='Ensemble', marker='*', linewidth=2, color='black')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Clustering Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    runExperiment()
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

from sklearn.preprocessing import RobustScaler

sys.path.append(r"src")

from algorithms.gmm_member import GMM_Simple
from algorithms.hierarchical_member import HierarchicalCentroidScratch
from algorithms.kmeans_member import kmeansScratch
from algorithms.silhoutte import silhouette_score

diabetesData = pd.read_csv(r"src\data\elliptical_data.csv")
# diabetesData = pd.read_csv("processed_diabetes_1000.csv")
xValues = diabetesData.values

robustScaler = RobustScaler()
xScaled = robustScaler.fit_transform(xValues)
# xScaled = xValues


ensembleScores = []
gmmScores = []
hierScores = []
kmScores = []
kRange = range(2, 21)

# print("Trọng số")
for k in kRange:
    gmmMember = GMM_Simple(k=k)
    gmmMember.fit(xScaled)
    hierMember = HierarchicalCentroidScratch(k=k)
    kmMember = kmeansScratch(k=k)

    #NHãn theo từng model đơn
    labelsGmm = gmmMember.predict(xScaled)
    labelsHier = hierMember.fit_predict(xScaled)
    labelsKm = kmMember.fit_predict(xScaled)

    sil_Gmm = max(silhouette_score(xScaled, labelsGmm), 1e-5)
    sil_Hier = max(silhouette_score(xScaled, labelsHier), 1e-5)
    sil_Km = max(silhouette_score(xScaled, labelsKm), 1e-5)

    gmmScores.append(sil_Gmm)
    hierScores.append(sil_Hier)
    kmScores.append(sil_Km)

    #weight --> lớn ah đến mx đồng thuận
    tong_Sil = sil_Gmm + sil_Hier + sil_Km
    weightGmm = sil_Gmm / tong_Sil
    weightHier = sil_Hier / tong_Sil
    weightKm = sil_Km / tong_Sil
    # print(f"{k:<5} | {weightGmm:<12.4f} | {weightHier:<12.4f} | {weightKm:<12.4f}") #xem wei

    # mx đồng thuật
    nSamples = xScaled.shape[0]
    weightedCoMatrix = np.zeros((nSamples, nSamples))

    weightedCoMatrix += weightGmm * (labelsGmm[:, np.newaxis] == labelsGmm)
    weightedCoMatrix += weightHier* (labelsHier[:, np.newaxis] == labelsHier)
    weightedCoMatrix += weightKm*(labelsKm[:, np.newaxis] == labelsKm)

    #--> K/c --> phân cụm mx đồng thuận
    distMatrix = 1 - weightedCoMatrix
    np.fill_diagonal(distMatrix, 0)
    finalEnsemble = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='complete')
    labelsEnsemble = finalEnsemble.fit_predict(distMatrix)

    ensembleScores.append(silhouette_score(xScaled, labelsEnsemble))


plt.figure(figsize=(14, 8))
#Ensemble, gmm, kmeans, hie
plt.plot(kRange, ensembleScores, 'o-', label='Weighted Ensemble', linewidth=3, color='red')
plt.plot(kRange, gmmScores, 's--', label='GMM', alpha=0.7)
plt.plot(kRange, hierScores, '^--', label='Hierarchical', alpha=0.7)
plt.plot(kRange, kmScores, 'x--', label='Kmeans', alpha=0.7)

plt.title("So sánh Silhoutte Ensemble - GMM, Hiera, Kmeans", fontsize=14)
plt.xlabel("k", fontsize=12)
plt.ylabel("Silhouette", fontsize=12)

plt.xticks(range(1, 21))
plt.yticks(np.arange(-0.25, 1.25, 0.25))
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
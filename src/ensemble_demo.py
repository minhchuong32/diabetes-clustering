import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

sys.path.append(r"src")

from algorithms.gmm_member import GMM_Simple
from algorithms.hierarchical_member import HierarchicalCentroidScratch
from algorithms.kmeans_member import kmeansScratch

# Load dữ liệu
diabetesData = pd.read_csv(r"src\data\elliptical_data.csv")
xValues = diabetesData.values

# Chuẩn hóa dữ liệu
robustScaler = RobustScaler()
xScaled = robustScaler.fit_transform(xValues)

ensembleScores = []
gmmScores = []
hierScores = []
kmScores = []
kRange = range(2, 21)

print("-" * 60)
print(f"{'k':<5} | {'GMM Weight':<12} | {'Hier Weight':<12} | {'KM Weight':<12}")
print("-" * 60)

for k in kRange:
    # Khởi tạo thuật toán
    gmmMember = GMM_Simple(k=k)
    hierMember = HierarchicalCentroidScratch(k=k)
    kmMember = kmeansScratch(k=k)

    # Dự đoán nhãn
    gmmFit = gmmMember.fit(xScaled)
    labelsGmm = gmmFit.predict(xScaled)
    labelsHier = hierMember.fit_predict(xScaled)
    labelsKm = kmMember.fit_predict(xScaled)

    # Tính điểm Silhouette cho từng thành viên
    silGmm = max(silhouette_score(xScaled, labelsGmm), 1e-5)
    silHier = max(silhouette_score(xScaled, labelsHier), 1e-5)
    silKm = max(silhouette_score(xScaled, labelsKm), 1e-5)

    gmmScores.append(silGmm)
    hierScores.append(silHier)
    kmScores.append(silKm)

    # Tính trọng số dựa trên Silhouette
    totalSil = silGmm + silHier + silKm
    weightGmm = silGmm / totalSil
    weightHier = silHier / totalSil
    weightKm = silKm / totalSil

    # In trọng số sau khi tính toán xong 1 k
    print(f"{k:<5} | {weightGmm:<12.4f} | {weightHier:<12.4f} | {weightKm:<12.4f}")

    # Xây dựng ma trận đồng thuận có trọng số
    nSamples = xScaled.shape[0]
    weightedCoMatrix = np.zeros((nSamples, nSamples))

    weightedCoMatrix += weightGmm * (labelsGmm[:, np.newaxis] == labelsGmm)
    weightedCoMatrix += weightHier * (labelsHier[:, np.newaxis] == labelsHier)
    weightedCoMatrix += weightKm * (labelsKm[:, np.newaxis] == labelsKm)

    # Chuyển đổi sang ma trận khoảng cách
    distMatrix = 1 - weightedCoMatrix
    np.fill_diagonal(distMatrix, 0)

    # Thực hiện Ensemble Clustering
    finalEnsemble = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='complete')
    labelsEnsemble = finalEnsemble.fit_predict(distMatrix)
    
    ensembleScores.append(silhouette_score(xScaled, labelsEnsemble))

# --- Trực quan hóa kết quả ---
plt.figure(figsize=(14, 8))

# Vẽ các đường biểu diễn
plt.plot(kRange, ensembleScores, 'o-', label='Weighted Ensemble', linewidth=3, color='red')
plt.plot(kRange, gmmScores, 's--', label='GMM', alpha=0.7)
plt.plot(kRange, hierScores, '^--', label='Hierarchical', alpha=0.7)
plt.plot(kRange, kmScores, 'x--', label='K-Means', alpha=0.7)

# Thiết lập tiêu đề và nhãn
plt.title("Weighted Ensemble vs Members Silhouette Comparison", fontsize=14)
plt.xlabel("Number of Clusters (k)", fontsize=12)
plt.ylabel("Silhouette Score", fontsize=12)

# Yêu cầu 1: Chuẩn hóa trục X là số tự nhiên từ 1 đến 20
plt.xticks(range(1, 21))

# Yêu cầu 2: Chuẩn hóa trục Y cách đều 0.25
# Silhouette score nằm trong khoảng [-1, 1], ta thiết lập dải hiển thị phù hợp
plt.yticks(np.arange(-0.25, 1.25, 0.25))

plt.legend(loc='best')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np
<<<<<<< HEAD
from src.algorithms.hierarchical_member import HierarchicalCentroidScratch

=======
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
>>>>>>> 4abe889a13c237916625f00a91974add22f99819

from sklearn.preprocessing import RobustScaler

<<<<<<< HEAD
    def fit_predict(self, labels_list):
        n_samples = len(labels_list[0])
        # Tạo ma trận biểu quyết NxN
        consensus_matrix = np.zeros((n_samples, n_samples))

        for labels in labels_list:
            for i in range(n_samples):
                for j in range(i, n_samples):
                    if labels[i] == labels[j]:
                        consensus_matrix[i, j] += 1
                        consensus_matrix[j, i] += 1

        # Lấy trung bình
        consensus_matrix /= len(labels_list)

        # Chuyển sang ma trận khoảng cách: Distance = 1 - Similarity
        distance_matrix = 1 - consensus_matrix

        # Chạy Hierarchical trên ma trận khoảng cách này
        # Lưu ý: Ta cần tùy chỉnh Hierarchical để nhận ma trận khoảng cách thay vì tọa độ X
        final_model = HierarchicalCentroidScratch(k=self.k)
        return self._cluster_from_dist(distance_matrix)
        # Ghi đè logic fit cho ma trận khoảng cách
        return self._cluster_from_dist(distance_matrix)

    def _cluster_from_dist(self, dist_matrix):
        n = len(dist_matrix)
        clusters = {i: [i] for i in range(n)}
        while len(clusters) > self.k:
            min_dist = np.inf
            to_merge = (0, 0)
            keys = list(clusters.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    d = np.mean(
                        dist_matrix[np.ix_(clusters[keys[i]], clusters[keys[j]])]
                    )
                    if d < min_dist:
                        min_dist = d
                        to_merge = (keys[i], keys[j])
            clusters[to_merge[0]].extend(clusters[to_merge[1]])
            del clusters[to_merge[1]]

        labels = np.zeros(n, dtype=int)
        for i, (k, points) in enumerate(clusters.items()):
            for p in points:
                labels[p] = i
        return labels
=======
sys.path.append(r"src")

from algorithms.gmm_member import GMM_Simple
from algorithms.hierarchical_member import HierarchicalCentroidScratch
from algorithms.kmeans_member import kmeansScratch
from algorithms.silhoutte import silhouette_score

def ensemble_scratch(xScaled, kRange=range(2, 21)):
    sil_kq = {
        'ensemble': [],
        'gmm': [],
        'hier': [],
        'kmeans': []
    }

    for k in kRange:
        gmmMember = GMM_Simple(k=k)
        gmmMember.fit(xScaled)
        hierMember = HierarchicalCentroidScratch(k=k)
        kmMember = kmeansScratch(k=k)

        labelsGmm = gmmMember.predict(xScaled)
        labelsHier = hierMember.fit_predict(xScaled)
        labelsKm = kmMember.fit_predict(xScaled)

        s_g = max(silhouette_score(xScaled, labelsGmm), 1e-5)
        s_h = max(silhouette_score(xScaled, labelsHier), 1e-5)
        s_k = max(silhouette_score(xScaled, labelsKm), 1e-5)

        sil_kq['gmm'].append(s_g)
        sil_kq['hier'].append(s_h)
        sil_kq['kmeans'].append(s_k)

        # Tính trọng số Adaptive
        total = s_g + s_h + s_k
        w_g, w_h, w_k = s_g/total, s_h/total, s_k/total

        #mx đồng thuận
        n = xScaled.shape[0]
        coMatrix = np.zeros((n, n))
        coMatrix += w_g * (labelsGmm[:, None] == labelsGmm)
        coMatrix += w_h * (labelsHier[:, None] == labelsHier)
        coMatrix += w_k * (labelsKm[:, None] == labelsKm)

        #ensemble mx khoang cách
        distMatrix = 1 - coMatrix
        np.fill_diagonal(distMatrix, 0)
        ensemble_model = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='complete')
        labelsEnsemble = ensemble_model.fit_predict(distMatrix)

        sil_kq['ensemble'].append(silhouette_score(xScaled, labelsEnsemble))

    return sil_kq

def ensemble_lib(xScaled, kRange=range(2, 21)):
    sil_kq = {'ensemble': [], 'gmm': [], 'hier': [], 'km': []}

    for k in kRange:
        # skleanr
        m_gmm = GaussianMixture(n_components=k, random_state=42).fit_predict(xScaled)
        Z = linkage(xScaled, method='centroid', metric = 'euclidean')
        m_hier = fcluster(Z, t=k, criterion='maxclust') - 1
        m_km = KMeans(n_clusters=k, random_state=42, n_init='auto').fit_predict(xScaled)

        s_g = max(silhouette_score(xScaled, m_gmm), 1e-5)
        s_h = max(silhouette_score(xScaled, m_hier), 1e-5)
        s_k = max(silhouette_score(xScaled, m_km), 1e-5)

        sil_kq['gmm'].append(s_g)
        sil_kq['hier'].append(s_h)
        sil_kq['km'].append(s_k)

        #mx đồng thuận
        w = np.array([s_g, s_h, s_k]) / (s_g + s_h + s_k)
        coM = (w[0]*(m_gmm[:,None]==m_gmm) + w[1]*(m_hier[:,None]==m_hier) + w[2]*(m_km[:,None]==m_km))

        labelsEns = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='complete').fit_predict(1 - coM)
        sil_kq['ensemble'].append(silhouette_score(xScaled, labelsEns))

    return sil_kq

def plot_comparison(results, kRange, title="So sánh Silhouette"):
    plt.figure(figsize=(12, 7))
    plt.plot(kRange, results['ensemble'], 'o-', label='Weighted Ensemble', color='red', linewidth=3)
    plt.plot(kRange, results['gmm'], 's--', label='GMM', alpha=0.6)
    plt.plot(kRange, results['hier'], '^--', label='Hierarchical', alpha=0.6)
    plt.plot(kRange, results['km'], 'x--', label='Kmeans', alpha=0.6)

    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.legend()
    plt.xticks(range(1, 21))
    plt.yticks(np.arange(-0.25, 1.25, 0.25))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()


# diabetesData = pd.read_csv("elliptical_data.csv")
# xValues = diabetesData.values
# robustScaler = RobustScaler()
# xScaled = robustScaler.fit_transform(xValues)
#
# results_scratch = ensemble_lib(xScaled, range(2, 21))
# plot_comparison(results_scratch, range(2, 21), "Kết quả của Scratch")
>>>>>>> 4abe889a13c237916625f00a91974add22f99819

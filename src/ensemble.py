import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from sklearn.preprocessing import RobustScaler

sys.path.append(r"src")

from algorithms.gmm_member import GaussianMixtureModel
from algorithms.hierarchical_member import HierarchicalCentroidScratch
from algorithms.kmeans_member import kmeansScratch
from algorithms.silhoutte import silhouette_score


def ensemble_lib_single(xScaled, k=3):
    m_gmm = GaussianMixture(n_components=k, random_state=42).fit_predict(xScaled)
    Z = linkage(xScaled, method="centroid", metric="euclidean")
    m_hier = fcluster(Z, t=k, criterion="maxclust") - 1
    m_km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(xScaled)

    #sill
    s_g = max(silhouette_score(xScaled, m_gmm), 1e-5)
    s_h = max(silhouette_score(xScaled, m_hier), 1e-5)
    s_k = max(silhouette_score(xScaled, m_km), 1e-5)

    w = np.array([s_g, s_h, s_k]) / (s_g + s_h + s_k)
    coM = (w[0] * (m_gmm[:, None] == m_gmm) +
           w[1] * (m_hier[:, None] == m_hier) +
           w[2] * (m_km[:, None] == m_km))

    labelsEns = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="complete").fit_predict(1 - coM)

    return labelsEns

def ensemble_scratch(xScaled, kRange=range(2, 21)):
    sil_kq = {"ensemble": [], "gmm": [], "hier": [], "kmeans": []}

    for k in kRange:
        gmmMember = GaussianMixtureModel(k=k)
        hierMember = HierarchicalCentroidScratch(k=k)
        kmMember = kmeansScratch(k=k)

        labelsGmm = gmmMember.fit_predict(xScaled)
        labelsHier = hierMember.fit_predict(xScaled)
        labelsKm = kmMember.fit_predict(xScaled)

        s_g = max(silhouette_score(xScaled, labelsGmm), 1e-5)
        s_h = max(silhouette_score(xScaled, labelsHier), 1e-5)
        s_k = max(silhouette_score(xScaled, labelsKm), 1e-5)

        sil_kq["gmm"].append(s_g)
        sil_kq["hier"].append(s_h)
        sil_kq["kmeans"].append(s_k)

        # Tính trọng số
        total = s_g + s_h + s_k
        w_g, w_h, w_k = s_g / total, s_h / total, s_k / total

        # mx đồng thuận
        n = xScaled.shape[0]
        coMatrix = np.zeros((n, n))
        coMatrix += w_g * (labelsGmm[:, None] == labelsGmm)
        coMatrix += w_h * (labelsHier[:, None] == labelsHier)
        coMatrix += w_k * (labelsKm[:, None] == labelsKm)

        # ensemble mx khoang cách
        distMatrix = 1 - coMatrix
        np.fill_diagonal(distMatrix, 0)
        ensemble_model = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="complete"
        )
        labelsEnsemble = ensemble_model.fit_predict(distMatrix)

        sil_kq["ensemble"].append(silhouette_score(xScaled, labelsEnsemble))

    return sil_kq


def ensemble_lib(xScaled, kRange=range(2, 21)):
    sil_kq = {"ensemble": [], "gmm": [], "hier": [], "km": []}

    for k in kRange:
        # skleanr
        m_gmm = GaussianMixture(n_components=k, random_state=42).fit_predict(xScaled)
        Z = linkage(xScaled, method="centroid", metric="euclidean")
        m_hier = fcluster(Z, t=k, criterion="maxclust") - 1
        m_km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(xScaled)

        s_g = max(silhouette_score(xScaled, m_gmm), 1e-5)
        s_h = max(silhouette_score(xScaled, m_hier), 1e-5)
        s_k = max(silhouette_score(xScaled, m_km), 1e-5)

        sil_kq["gmm"].append(s_g)
        sil_kq["hier"].append(s_h)
        sil_kq["km"].append(s_k)

        # mx đồng thuận
        w = np.array([s_g, s_h, s_k]) / (s_g + s_h + s_k)
        coM = (
            w[0] * (m_gmm[:, None] == m_gmm)
            + w[1] * (m_hier[:, None] == m_hier)
            + w[2] * (m_km[:, None] == m_km)
        )

        labelsEns = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="complete"
        ).fit_predict(1 - coM)
        sil_kq["ensemble"].append(silhouette_score(xScaled, labelsEns))

    return sil_kq


def plot_comparison(results, kRange, title="So sánh Silhouette"):
    plt.figure(figsize=(12, 7))
    plt.plot(
        kRange,
        results["ensemble"],
        "o-",
        label="Weighted Ensemble",
        color="red",
        linewidth=3,
    )
    plt.plot(kRange, results["gmm"], "s--", label="GMM", alpha=0.6)
    plt.plot(kRange, results["hier"], "^--", label="Hierarchical", alpha=0.6)
    plt.plot(kRange, results["km"], "x--", label="Kmeans", alpha=0.6)

    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.legend()
    plt.xticks(range(1, 21))
    plt.yticks(np.arange(-0.25, 1.25, 0.25))
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.show()


# diabetesData = pd.read_csv("elliptical_data.csv")
# xValues = diabetesData.values
# robustScaler = RobustScaler()
# xScaled = robustScaler.fit_transform(xValues)
#
# results_scratch = ensemble_lib(xScaled, range(2, 21))
# plot_comparison(results_scratch, range(2, 21), "Kết quả của Scratch")

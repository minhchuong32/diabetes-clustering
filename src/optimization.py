import scipy.stats as st
import pandas as pd
import numpy as np
import sys

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import RobustScaler

sys.path.append(r"src")

from algorithms.gmm_member import GMM_Simple
from algorithms.hierarchical_member import HierarchicalCentroidScratch
from algorithms.kmeans_member import kmeansScratch
from algorithms.silhoutte import silhouette_score



def analysis(data_path, k=2):

    diabetesData = pd.read_csv(data_path)
    xValues = diabetesData.values

    robustScaler = RobustScaler()
    xScaled = robustScaler.fit_transform(xValues)

    cols = diabetesData.columns.tolist()
    df_profile = diabetesData[cols].copy()

    #3 model
    gmm = GMM_Simple(k)
    gmm.fit(xScaled)
    hier = HierarchicalCentroidScratch(k=k)
    km = kmeansScratch(k=k)

    labelsG = gmm.predict(xScaled)
    labelsH = hier.fit_predict(xScaled)
    labelsK = km.fit_predict(xScaled)

    #silou
    sil_Gmm = silhouette_score(xScaled, labelsG)
    sil_Hier = silhouette_score(xScaled, labelsH)
    sil_Km = silhouette_score(xScaled, labelsK)

    tong_Sil = max(sil_Gmm + sil_Hier + sil_Km, 1e-6)
    weightGmm = sil_Gmm / tong_Sil
    weightHier = sil_Hier / tong_Sil
    weightKm = sil_Km / tong_Sil

    #mx dododngf thuận
    n = xScaled.shape[0]
    C = np.zeros((n, n))
    C += weightGmm * (labelsG[:, None] == labelsG)
    C += weightHier * (labelsH[:, None] == labelsH)
    C += weightKm * (labelsK[:, None] == labelsK)

    #--> kc
    Dis_mx = 1 - C
    ensemble = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="complete")
    labelsEnsemble = ensemble.fit_predict(Dis_mx)
    df_profile["Cluster"] = labelsEnsemble

    #TK các chí số theo cụm
    def cluster_stats(df, cluster_col="Cluster"):
        rows = []
        for var in cols:
            for cluster in sorted(df[cluster_col].unique()):
                data = df[df[cluster_col] == cluster][var].dropna()
                n_samp = len(data)

                if n_samp < 2:
                    continue

                mean = data.mean()
                sd = data.std(ddof=1)
                se = sd / np.sqrt(n_samp)

                # 95%
                ci_low, ci_high = st.t.interval(0.95, n_samp-1, loc=mean, scale=se)

                rows.append({
                    "Variable": var,
                    "Cluster": f"Cluster {cluster+1}",
                    "N": n_samp,
                    "Mean": mean,
                    "SD": sd,
                    "SE": se,
                    "95% CI LL": ci_low,
                    "95% CI UL": ci_high,
                    "Min": data.min(),
                    "Max": data.max()
                })
        return pd.DataFrame(rows)

    stats_table = cluster_stats(df_profile)

    return stats_table.sort_values(["Variable", "Cluster"])

# res_table = analysis(r"src\data\elliptical_data.csv", k=2)
# print(res_table)
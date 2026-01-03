import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def gaussian_pdf(X, mean, cov):
    d = X.shape[1]
    cov = cov + np.eye(d) * 1e-6  # chống suy biến

    diff = X - mean
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)

    coef = 1.0 / np.sqrt((2 * np.pi) ** d * det_cov)
    expo = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)

    return coef * np.exp(expo)


class GaussianMixtureModel:
    def __init__(self, k=2, max_iter=50):
        self.k = k
        self.max_iter = max_iter
        self.eps = 1e-6

    def fit_predict(self, X):
        n, d = X.shape

        self.weights = np.ones(self.k) / self.k
        idx = np.random.choice(n, self.k, replace=False)
        self.means = X[idx]
        self.covs = np.array([np.eye(d) for _ in range(self.k)])

        for _ in range(self.max_iter):
            # E-step
            resp = np.zeros((n, self.k))
            for k in range(self.k):
                resp[:, k] = self.weights[k] * gaussian_pdf(
                    X, self.means[k], self.covs[k]
                )

            row_sum = np.sum(resp, axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1e-10  # chống chia 0
            resp /= row_sum

            # M-step
            Nk = np.sum(resp, axis=0)

            for k in range(self.k):
                if Nk[k] == 0:  # cụm rỗng
                    continue

                self.means[k] = np.sum(resp[:, k : k + 1] * X, axis=0) / Nk[k]
                diff = X - self.means[k]
                self.covs[k] = (resp[:, k] * diff.T) @ diff / Nk[k]
                self.covs[k] += np.eye(d) * self.eps

            self.weights = Nk / n

        return np.argmax(resp, axis=1)


def silhouette_score(X, labels):
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0  # không xác định

    s = []
    for i in range(len(X)):
        same = X[labels == labels[i]]
        other_clusters = [X[labels == k] for k in unique_labels if k != labels[i]]

        if len(other_clusters) == 0:
            continue

        a = np.mean(np.linalg.norm(same - X[i], axis=1))
        b = min(np.mean(np.linalg.norm(c - X[i], axis=1)) for c in other_clusters)

        s.append((b - a) / max(a, b))

    return np.mean(s)


def plot_silhouette(X, k_min=2, k_max=10):
    scores = []

    for k in range(k_min, k_max + 1):
        model = GaussianMixtureModel(k)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)
        print(f"K = {k} | Silhouette = {score:.4f}")

    plt.plot(range(k_min, k_max + 1), scores, marker="o")
    plt.xlabel("Số cụm K")
    plt.ylabel("Silhouette Score")
    plt.title("Đánh giá số cụm tối ưu bằng GMM")
    plt.grid()
    plt.show()


df = pd.read_csv("processed_diabetes_1000.csv")
data_columns = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]
X = df[data_columns].values.astype(float)
plot_silhouette(X)

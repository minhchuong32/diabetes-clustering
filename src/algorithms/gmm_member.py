import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import silhouette_score


class GaussianMixtureModel:
    def __init__(self, k=2, max_iters=50):
        self.k = k
        self.max_iters = max_iters
        self.tol = 1e-6
        self.eps = 1e-6  # Chống lỗi ma trận suy biến

    def khoi_tao_tham_so(self, X):
        """Khởi tạo Trọng số, Tâm cụm và Hiệp phương sai"""
        self.n_samples, self.n_features = X.shape
        self.weights = np.full(self.k, 1.0 / self.k)
        indices = np.random.choice(self.n_samples, self.k, replace=False)
        self.means = X[indices]
        self.covs = np.array([np.eye(self.n_features) for _ in range(self.k)])

    def buoc_E(self, X):
        """Bước Expectation: Tính xác suất trách nhiệm"""
        trach_nhiem = np.zeros((self.n_samples, self.k))
        for k in range(self.k):
            pdf = multivariate_normal.pdf(
                X, mean=self.means[k], cov=self.covs[k], allow_singular=True
            )
            trach_nhiem[:, k] = self.weights[k] * pdf

        sum_row = trach_nhiem.sum(axis=1, keepdims=True)
        return trach_nhiem / (sum_row + 1e-10)

    def buoc_M(self, X, trach_nhiem):
        """Bước Maximization: Cập nhật tham số model"""
        Nk = trach_nhiem.sum(axis=0)
        self.weights = Nk / self.n_samples

        for k in range(self.k):
            self.means[k] = np.sum(trach_nhiem[:, [k]] * X, axis=0) / Nk[k]
            diff = X - self.means[k]
            self.covs[k] = np.dot((trach_nhiem[:, k] * diff.T), diff) / Nk[k]
            self.covs[k] += np.eye(self.n_features) * self.eps

    def fit_predict(self, X):
        self.khoi_tao_tham_so(X)

        for i in range(self.max_iters):
            old_means = self.means.copy()

            # Chu trình EM
            trach_nhiem = self.buoc_E(X)
            self.buoc_M(X, trach_nhiem)

            # Kiểm tra hội tụ
            if np.linalg.norm(self.means - old_means) < self.tol:
                break

        # Sau khi fit xong, lấy xác suất cuối cùng để gán nhãn
        final_trach_nhiem = self.buoc_E(X)
        return np.argmax(final_trach_nhiem, axis=1)


# df = pd.read_csv('processed_diabetes_1000.csv')
# data_columns = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
#                 'num_medications', 'number_outpatient', 'number_emergency',
#                 'number_inpatient', 'number_diagnoses']
# X = df[data_columns].values.astype(float)


# def ve_do_thi_silhouette(X, k_min=2, k_max=10):
#     ds_silhouette = []
#     ds_k = range(k_min, k_max + 1)

#     print("Đang tính toán Silhouette cho từng K...")
#     for k in ds_k:
#         model = GMM_Simple(k=k)
#         nhan = model.fit_predict(X)

#         score = silhouette_score(X, nhan)
#         ds_silhouette.append(score)
#         print(f"K = {k} | Silhouette Score: {score:.4f}")

#     plt.figure(figsize=(10, 5))
#     plt.plot(ds_k, ds_silhouette, marker="o", linestyle="--", color="b")
#     plt.title("Đánh giá số cụm tối ưu bằng Silhouette Score (GMM)")
#     plt.xlabel("Số lượng cụm (K)")
#     plt.ylabel("Silhouette Score")
#     plt.grid(True)
#     plt.show()


# ve_do_thi_silhouette(X)

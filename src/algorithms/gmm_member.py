import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal  # Phân phối chuẩn đa biến
from sklearn.metrics import silhouette_score  # Đánh giá phân cụm
import pandas as pd


class GaussianMixtureModel:
    def __init__(self, k=2, max_iters=50):
        self.k = k  # Số cụm (Gaussian components)
        self.max_iters = max_iters  # Số vòng lặp EM tối đa
        self.tol = 1e-6  # Ngưỡng hội tụ
        self.eps = 1e-6  # Tránh ma trận hiệp phương sai suy biến

    def khoi_tao_tham_so(self, X):
        self.N, self.D = X.shape  # Số mẫu và số đặc trưng
        self.W = np.full(self.k, 1 / self.k)  # Trọng số ban đầu π_k
        idx = np.random.choice(
            self.N, self.k, replace=False
        )  # Chọn ngẫu nhiên K điểm làm mean ban đầu
        self.means = X[idx]  # khoi tao mean
        self.covs = np.array([np.eye(self.D)] * self.k)  # khoi tao covarience

    def buoc_E(self, X):
        # khoi tao ma tran trach nhiem NxK
        trach_nhiem = np.zeros((self.N, self.k))
        for k in range(self.k):
            # Tính mật độ xác suất Gaussian
            pdf = multivariate_normal.pdf(
                X,
                mean=self.means[k],
                cov=self.covs[k],
                allow_singular=True,  # Tránh ma trận hiệp phương sai suy biến (det=0)
            )
            # Nhân với trọng số π_k
            trach_nhiem[:, k] = self.W[k] * pdf
        # tính tổng các xs trách nhiệm theo từng hàng
        sum_row = trach_nhiem.sum(axis=1, keepdims=True)
        return trach_nhiem / (sum_row + 1e-10)

    def buoc_M(self, X, trach_nhiem):
        # Tổng xs trách nhiệm cho mỗi cụm
        Nk = trach_nhiem.sum(axis=0)
        # Cập nhật trọng số
        self.W = Nk / self.N
        for k in range(self.k):
            # Cập nhật mean : means[k] = Σ trach_nhiem[i,k] * x_i / Nk[k]
            self.means[k] = np.sum(trach_nhiem[:, [k]] * X, axis=0) / Nk[k]
            # Cập nhật covariance: covs[k] = Σ trach_nhiem[i,k] * (x_i - mean_k)(x_i - mean_k)^T / Nk[k]
            self.covs[k] = (
                np.dot((trach_nhiem[:, k] * (X - self.means[k]).T), X - self.means[k])
                / Nk[k]
            )
            # Cộng eps để tránh suy biến: covs = covs + eps*I
            self.covs[k] += np.eye(self.D) * self.eps

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

        # Gán nhãn theo xác suất cao nhất
        final_trach_nhiem = self.buoc_E(X)
        return np.argmax(final_trach_nhiem, axis=1)


df = pd.read_csv("src/data/processed_diabetes_1000.csv")

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

# Chuyển dữ liệu sang numpy array
X = df[data_columns].values.astype(float)


def ve_do_thi_silhouette(X, k_min=2, k_max=10):
    ds_silhouette = []
    ds_k = range(k_min, k_max + 1)

    print("Đang tính toán Silhouette cho từng K...")

    for k in ds_k:
        model = GaussianMixtureModel(k=k)
        nhan = model.fit_predict(X)

        # Tính Silhouette Score
        score = silhouette_score(X, nhan)
        ds_silhouette.append(score)

        print(f"K = {k} | Silhouette Score: {score:.4f}")

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 5))
    plt.plot(ds_k, ds_silhouette, marker="o", linestyle="--")
    plt.title("Đánh giá số cụm tối ưu bằng Silhouette Score (GaussianMixtureModel)")
    plt.xlabel("Số lượng cụm (K)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.show()


ve_do_thi_silhouette(X)

import numpy as np                         
import matplotlib.pyplot as plt             
from scipy.stats import multivariate_normal # Phân phối chuẩn đa biến
from sklearn.metrics import silhouette_score # Đánh giá phân cụm
import pandas as pd                        

class GaussianMixtureModel:
    def __init__(self, k=2, max_iters=50):
        self.k = k                          # Số cụm (Gaussian components)
        self.max_iters = max_iters          # Số vòng lặp EM tối đa
        self.tol = 1e-6                     # Ngưỡng hội tụ
        self.eps = 1e-6                     # Tránh ma trận hiệp phương sai suy biến

    def khoi_tao_tham_so(self, X):
        """
        Khởi tạo:
        - Trọng số π_k
        - Trung bình μ_k
        - Hiệp phương sai Σ_k
        """
        self.n_samples, self.n_features = X.shape

        # Trọng số ban đầu: chia đều cho các cụm
        self.weights = np.full(self.k, 1.0 / self.k)

        # Chọn ngẫu nhiên k điểm trong dữ liệu làm mean ban đầu
        indices = np.random.choice(self.n_samples, self.k, replace=False)
        self.means = X[indices]

        # Khởi tạo hiệp phương sai là ma trận đơn vị
        self.covs = np.array([
            np.eye(self.n_features) for _ in range(self.k)
        ])

    def buoc_E(self, X):
        """
        Expectation step:
        Tính xác suất trách nhiệm γ(z_k | x_i)
        """
        # Ma trận trách nhiệm (n_samples x k)
        trach_nhiem = np.zeros((self.n_samples, self.k))

        for k in range(self.k):
            # Tính mật độ xác suất Gaussian
            pdf = multivariate_normal.pdf(
                X,
                mean=self.means[k],
                cov=self.covs[k],
                allow_singular=True
            )

            # Nhân với trọng số π_k
            trach_nhiem[:, k] = self.weights[k] * pdf

        # Chuẩn hóa để tổng mỗi hàng = 1
        sum_row = trach_nhiem.sum(axis=1, keepdims=True)

        return trach_nhiem / (sum_row + 1e-10)

    def buoc_M(self, X, trach_nhiem):
        """
        Maximization step:
        Cập nhật π_k, μ_k, Σ_k
        """
        # Tổng trách nhiệm cho mỗi cụm
        Nk = trach_nhiem.sum(axis=0)

        # Cập nhật trọng số
        self.weights = Nk / self.n_samples

        for k in range(self.k):
            # Cập nhật mean
            self.means[k] = np.sum(
                trach_nhiem[:, [k]] * X,
                axis=0
            ) / Nk[k]

            # Cập nhật covariance
            diff = X - self.means[k]
            self.covs[k] = np.dot(
                (trach_nhiem[:, k] * diff.T),
                diff
            ) / Nk[k]

            # Cộng eps để tránh suy biến
            self.covs[k] += np.eye(self.n_features) * self.eps

    def fit_predict(self, X):
        # Khởi tạo tham số
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


df = pd.read_csv('processed_diabetes_1000.csv')

data_columns = [
    'time_in_hospital',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_outpatient',
    'number_emergency',
    'number_inpatient',
    'number_diagnoses'
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
    plt.plot(ds_k, ds_silhouette, marker='o', linestyle='--')
    plt.title("Đánh giá số cụm tối ưu bằng Silhouette Score (GaussianMixtureModel)")
    plt.xlabel("Số lượng cụm (K)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.show()


ve_do_thi_silhouette(X)

import numpy as np


class GMMScratch:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def multivariate_gaussian(self, X, mean, cov):
        """Tính hàm mật độ xác suất của phân phối chuẩn đa biến."""
        n = X.shape[1]
        diff = X - mean
        # Thêm một lượng nhỏ vào đường chéo ma trận hiệp phương sai để tránh ma trận suy biến (singular)
        cov = cov + np.eye(n) * 1e-6

        det = np.linalg.det(cov)
        inv = np.linalg.inv(cov)

        # Công thức PDF của Gaussian
        exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)
        return (1.0 / np.sqrt((2 * np.pi) ** n * det)) * np.exp(exponent)

    def fit_predict(self, X):
        n_samples, n_features = X.shape

        # 1. Khởi tạo các tham số
        # Trọng số của các cụm (Weights)
        self.weights = np.full(self.k, 1 / self.k)
        # Tâm của các cụm (Means) - lấy ngẫu nhiên từ dữ liệu
        idx = np.random.choice(n_samples, self.k, replace=False)
        self.means = X[idx]
        # Ma trận hiệp phương sai (Covariances) - khởi tạo là ma trận đơn vị
        self.covs = [np.eye(n_features) for _ in range(self.k)]

        # Ma trận Responsibilities (xác suất điểm i thuộc cụm j)
        resp = np.zeros((n_samples, self.k))

        for i in range(self.max_iters):
            prev_means = self.means.copy()

            # --- Bước E (Expectation): Tính xác suất ---
            for j in range(self.k):
                resp[:, j] = self.weights[j] * self.multivariate_gaussian(
                    X, self.means[j], self.covs[j]
                )

            # Chuẩn hóa để tổng xác suất mỗi dòng = 1
            sum_resp = resp.sum(axis=1, keepdims=True)
            # Tránh chia cho 0
            sum_resp[sum_resp == 0] = 1e-10
            resp /= sum_resp

            # --- Bước M (Maximization): Cập nhật tham số ---
            N_j = resp.sum(axis=0)  # Tổng xác suất của mỗi cụm

            for j in range(self.k):
                # Cập nhật Means
                self.means[j] = (resp[:, j].reshape(-1, 1) * X).sum(axis=0) / N_j[j]

                # Cập nhật Covariances
                diff = X - self.means[j]
                self.covs[j] = (resp[:, j].reshape(-1, 1) * diff).T @ diff / N_j[j]

                # Cập nhật Weights
                self.weights[j] = N_j[j] / n_samples

            # Kiểm tra hội tụ
            if np.linalg.norm(self.means - prev_means) < self.tol:
                break

        # Trả về nhãn là cụm có xác suất cao nhất
        return np.argmax(resp, axis=1)

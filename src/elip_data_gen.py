import pandas as pd
import numpy as np
import os
from sklearn.datasets import make_blobs

# 1. Định nghĩa và tạo thư mục lưu trữ
dirPath = "src/data"
if not os.path.exists(dirPath):
    os.makedirs(dirPath)

# 2. Sinh dữ liệu gốc (hình cầu)
nSamples = 1000
xRaw, yTrue = make_blobs(n_samples=nSamples, centers=3, cluster_std=1.0, random_state=42)

# 3. Tạo ma trận biến đổi để kéo giãn dữ liệu thành hình elip
# Ma trận này sẽ làm lệch các trục tọa độ
transformationMatrix = [[0.60, -0.60], [-0.35, 0.85]]
xElliptical = np.dot(xRaw, transformationMatrix)

# 4. Đưa vào DataFrame
dfElliptical = pd.DataFrame(xElliptical, columns=["Feature1", "Feature2"])
dfElliptical["Target"] = yTrue # Lưu nhãn gốc để đối chiếu nếu cần

# 5. Lưu xuống file CSV
fileName = "elliptical_data.csv"
savePath = os.path.join(dirPath, fileName)
dfElliptical.to_csv(savePath, index=False)

print(f"--- Đã tạo dữ liệu thành công ---")
print(f"Vị trí lưu: {savePath}")
print(f"Số mẫu: {nSamples}")
print(f"Hình dạng cụm: Elip (Anisotropic)")
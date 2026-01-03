import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

def getEnsembleScore(xScaled, labelsList, weightsList, k):
    nSamples = xScaled.shape[0]
    weightedCoMatrix = np.zeros((nSamples, nSamples))
    
    # Chuẩn hóa trọng số
    totalW = sum(weightsList)
    normWeights = [w / totalW for w in weightsList]
    
    # Xây dựng ma trận đồng thuận (Broadcasting)
    for labels, w in zip(labelsList, normWeights):
        weightedCoMatrix += w * (labels[:, np.newaxis] == labels)
    
    distMatrix = 1 - weightedCoMatrix
    np.fill_diagonal(distMatrix, 0)
    
    # Gom cụm cuối cùng (Complete Linkage)
    finalClustering = AgglomerativeClustering(
        n_clusters=k, 
        metric='precomputed', 
        linkage='complete'
    )
    yEns = finalClustering.fit_predict(distMatrix)
    
    if len(np.unique(yEns)) > 1:
        return silhouette_score(xScaled, yEns)
    return -1.0

def runAblationStudy():
    filePath = os.path.join('src', 'data', 'elliptical_data.csv')
    if not os.path.exists(filePath):
        print("Không tìm thấy file dữ liệu.")
        return
    
    diabetesData = pd.read_csv(filePath)
    xScaled = RobustScaler().fit_transform(diabetesData.values)
    
    kRange = list(range(2, 21))
    
    # Khởi tạo các danh sách lưu điểm số cho các biến thể
    results = {
        'Full_Ensemble': [],    # Đầy đủ 3 thuật toán + Trọng số
        'No_GMM': [],           # Biến thể 1: Chỉ KM + Hier
        'No_KMeans': [],        # Biến thể 2: Chỉ GMM + Hier
        'No_Hierarchical': [],  # Biến thể 3: Chỉ GMM + KM
        'Unweighted': []        # Biến thể 4: Đầy đủ nhưng trọng số bằng nhau (1/3)
    }

    for k in kRange:
        # 1. Chạy các mô hình thành viên tối ưu
        kmModel = KMeans(n_clusters=k, n_init='auto', random_state=42)
        lKm = kmModel.fit_predict(xScaled)
        
        gmmModel = GaussianMixture(n_components=k, random_state=42)
        lGmm = gmmModel.fit(xScaled).predict(xScaled)
        
        hierModel = AgglomerativeClustering(n_clusters=k, linkage='complete')
        lHier = hierModel.fit_predict(xScaled)

        # 2. Tính điểm Silhouette làm trọng số
        sKm = max(silhouette_score(xScaled, lKm), 1e-5)
        sGmm = max(silhouette_score(xScaled, lGmm), 1e-5)
        sHier = max(silhouette_score(xScaled, lHier), 1e-5)
        
        # 3. Tính toán Full Model và 4 Biến thể Ablation
        # Full Ensemble
        results['Full_Ensemble'].append(getEnsembleScore(xScaled, [lGmm, lHier, lKm], [sGmm, sHier, sKm], k))
        
        # Biến thể 1: Không GMM
        results['No_GMM'].append(getEnsembleScore(xScaled, [lHier, lKm], [sHier, sKm], k))
        
        # Biến thể 2: Không KMeans
        results['No_KMeans'].append(getEnsembleScore(xScaled, [lGmm, lHier], [sGmm, sHier], k))
        
        # Biến thể 3: Không Hierarchical
        results['No_Hierarchical'].append(getEnsembleScore(xScaled, [lGmm, lKm], [sGmm, sKm], k))
        
        # Biến thể 4: Không trọng số (Trọng số bằng nhau)
        results['Unweighted'].append(getEnsembleScore(xScaled, [lGmm, lHier, lKm], [1.0, 1.0, 1.0], k))
        
        print(f"Hoàn thành k={k}")

    # --- Lưu kết quả CSV ---
    results['k'] = kRange
    pd.DataFrame(results).to_csv('ellip_ablation_study_results.csv', index=False, encoding='utf-8-sig')

    # --- Trực quan hóa kết quả Ablation Study ---
    plt.figure(figsize=(14, 8))
    
    # Đường chính: Full Ensemble
    plt.plot(kRange, results['Full_Ensemble'], 'o-', label='Full Weighted Ensemble', linewidth=4, color='red')
    
    # Các biến thể lược bỏ thành phần (Ablation)
    plt.plot(kRange, results['No_GMM'], '--', label='BT1: Không GMM', alpha=0.8, color='blue', marker='s')
    plt.plot(kRange, results['No_KMeans'], '--', label='BT2: Không KMeans', alpha=0.8, color='green', marker='^')
    plt.plot(kRange, results['No_Hierarchical'], '--', label='BT3: Không Hierarchical', alpha=0.8, color='orange', marker='d')
    
    # Biến thể lược bỏ cơ chế trọng số
    plt.plot(kRange, results['Unweighted'], 'k:', label='BT4: Unweighted (Equal Weights)', linewidth=2.5)
    
    plt.title("Nghiên cứu lược bỏ (Ablation Study): Đánh giá tầm quan trọng của từng thành phần", fontsize=14)
    plt.xlabel("Số cụm (k)", fontsize=12)
    plt.ylabel("Silhouette Score", fontsize=12)
    plt.xticks(kRange)
    plt.yticks(np.arange(-0.25, 1.05, 0.25))
    plt.legend(loc='upper right', shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    runAblationStudy()
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import các thuật toán từ thư mục src
from src.algorithms.kmeans_member import KMeansScratch
from src.algorithms.hierarchical_member import HierarchicalScratch
from src.algorithms.gmm_member import GMMScratch
from src.ensemble import EnsembleClustering

# Thiết lập giao diện
ctk.set_appearance_mode("System")  # Chế độ sáng/tối theo Windows
ctk.set_default_color_theme("blue")

class ModernDiabetesApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Hệ Thống Phân Cụm Bệnh Nhân Tiểu Đường - Ensemble Model")
        self.geometry("1100x600")

        # Cấu hình Layout Grid (Sidebar + Main)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- SIDEBAR ---
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="DIABETES\nANALYSIS", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.pack(pady=20)

        self.label_k = ctk.CTkLabel(self.sidebar, text="Số lượng cụm (k):")
        self.label_k.pack(pady=(10, 0))
        self.entry_k = ctk.CTkEntry(self.sidebar, placeholder_text="Ví dụ: 3")
        self.entry_k.insert(0, "3")
        self.entry_k.pack(pady=10, padx=20)

        self.btn_load = ctk.CTkButton(self.sidebar, text="Chọn file CSV", command=self.load_data)
        self.btn_load.pack(pady=10, padx=20)

        self.btn_run = ctk.CTkButton(self.sidebar, text="Chạy Ensemble", fg_color="green", hover_color="darkgreen", command=self.run_clustering)
        self.btn_run.pack(pady=10, padx=20)

        # --- MAIN CONTENT ---
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.status_label = ctk.CTkLabel(self.main_frame, text="Vui lòng nạp dữ liệu để bắt đầu", font=ctk.CTkFont(size=14))
        self.status_label.pack(pady=10)

        # Khu vực vẽ biểu đồ
        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        self.data = None

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.status_label.configure(text=f"Đã nạp: {file_path.split('/')[-1]} ({len(self.data)} dòng)", text_color="cyan")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể đọc file: {e}")

    def run_clustering(self):
        if self.data is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn file dữ liệu trước!")
            return

        try:
            k = int(self.entry_k.get())
            # Lấy toàn bộ dữ liệu (bỏ qua cột label nếu có)
            X = self.data.select_dtypes(include=[np.number]).values
            
            # Để tránh treo máy với code-tay NxN, giới hạn dữ liệu hiển thị
            X_subset = X[:150] 
            
            self.status_label.configure(text="Đang tính toán biểu quyết (Ensemble)...", text_color="yellow")
            self.update()

            # 1. Chạy 3 thuật toán "code tay"
            km = KMeansScratch(k=k)
            labels_km = km.fit_predict(X_subset)

            gmm = GMMScratch(k=k)
            labels_gmm = gmm.fit_predict(X_subset)

            hi = HierarchicalScratch(k=k)
            labels_hi = hi.fit_predict(X_subset)

            # 2. Ensemble - Biểu quyết tạo ma trận khoảng cách
            ensemble = EnsembleClustering(k=k)
            final_labels = ensemble.fit_predict([labels_km, labels_gmm, labels_hi])

            # 3. Vẽ kết quả lên đồ thị (Dùng 2 chiều Glucose và BMI để minh họa)
            self.update_chart(X_subset, final_labels)
            
            self.status_label.configure(text=f"Hoàn thành! Đã phân thành {k} cụm.", text_color="lime")
            
        except ValueError:
            messagebox.showerror("Lỗi", "Số cụm k phải là số nguyên!")
        except Exception as e:
            messagebox.showerror("Lỗi hệ thống", str(e))

    def update_chart(self, X, labels):
        self.ax.clear()
        # Giả sử cột 1 là Glucose, cột 5 là BMI
        scatter = self.ax.scatter(X[:, 1], X[:, 5], c=labels, cmap='viridis', edgecolors='white')
        self.ax.set_title("Kết quả Phân cụm Ensemble (Glucose vs BMI)")
        self.ax.set_xlabel("Glucose")
        self.ax.set_ylabel("BMI")
        self.fig.colorbar(scatter, ax=self.ax)
        self.canvas.draw()

if __name__ == "__main__":
    app = ModernDiabetesApp()
    app.mainloop()
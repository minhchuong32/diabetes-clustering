import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import sys
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.preprocessing import RobustScaler

# Thiết lập đường dẫn để import từ thư mục src
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import đúng tên Class từ các file algorithms của bạn
from algorithms.gmm_member import GaussianMixtureModel
from algorithms.hierarchical_member import HierarchicalCentroidScratch
from algorithms.kmeans_member import kmeansScratch
from algorithms.silhoutte import silhouette_score
from ensemble import EnsembleClustering


class DiabetesClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống Phân cụm Bệnh Tiểu đường - Ensemble Learning")
        self.root.geometry("1200x850")
        self.root.configure(bg="#f8fafc")

        # Biến lưu trữ dữ liệu
        self.df = None
        self.X_scaled = None
        self.ensemble_labels = None
        self.selected_k = 3
        self.silhouette_scores = []
        self.k_values = []

        self.setup_styles()
        self.create_widgets()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#f8fafc", padding=5)
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[15, 5])
        style.configure("Action.TButton", font=("Segoe UI", 11, "bold"))

    def create_widgets(self):
        # Header
        header = tk.Frame(self.root, bg="#4f46e5", height=70)
        header.pack(fill="x")
        tk.Label(
            header,
            text="🏥 DIABETES PATIENT CLUSTERING SYSTEM",
            font=("Segoe UI", 18, "bold"),
            bg="#4f46e5",
            fg="white",
        ).pack(pady=15)

        # Main Notebook
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=15, pady=10)

        # Các Tabs
        self.tab_data = tk.Frame(self.nb, bg="white")
        self.tab_elbow = tk.Frame(self.nb, bg="white")
        self.tab_analysis = tk.Frame(self.nb, bg="white")
        self.tab_predict = tk.Frame(self.nb, bg="white")

        self.nb.add(self.tab_data, text=" 📂 TẢI DỮ LIỆU ")
        self.nb.add(self.tab_elbow, text=" 📊 TỐI ƯU K ")
        self.nb.add(self.tab_analysis, text=" 📈 PHÂN TÍCH NHÓM ")
        self.nb.add(self.tab_predict, text=" 🔮 DỰ ĐOÁN MỚI ")

        self.setup_tab_data()
        self.setup_tab_elbow()
        self.setup_tab_analysis()
        self.setup_tab_predict()

    # --- TAB 1: QUẢN LÝ DỮ LIỆU ---
    def setup_tab_data(self):
        center_frame = tk.Frame(self.tab_data, bg="white")
        center_frame.place(relx=0.5, rely=0.4, anchor="center")

        tk.Label(
            center_frame,
            text="Bắt đầu bằng cách tải file dữ liệu tiểu đường (.csv)",
            font=("Segoe UI", 12),
            bg="white",
            fg="#64748b",
        ).pack(pady=10)

        tk.Button(
            center_frame,
            text="📂 TẢI FILE CSV",
            command=self.load_csv,
            bg="#6366f1",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            padx=30,
            pady=12,
            relief="flat",
            cursor="hand2",
        ).pack(pady=10)

        self.btn_run = tk.Button(
            center_frame,
            text="▶️ CHẠY THUẬT TOÁN ENSEMBLE",
            command=self.run_clustering_process,
            bg="#10b981",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            padx=30,
            pady=12,
            relief="flat",
            state="disabled",
            cursor="hand2",
        )
        self.btn_run.pack(pady=10)

        self.lbl_info = tk.Label(
            center_frame,
            text="Chưa có dữ liệu nào được tải",
            font=("Segoe UI", 10),
            bg="white",
            fg="#ef4444",
        )
        self.lbl_info.pack(pady=10)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                # Lọc lấy các cột số
                data_numeric = self.df.select_dtypes(include=[np.number])
                scaler = RobustScaler()
                self.X_scaled = scaler.fit_transform(data_numeric)

                self.lbl_info.config(
                    text=f"✅ Đã tải: {file_path.split('/')[-1]} ({len(self.df)} dòng)",
                    fg="#10b981",
                )
                self.btn_run.config(state="normal")
                messagebox.showinfo("Thành công", "Dữ liệu đã sẵn sàng để phân cụm!")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể đọc file: {str(e)}")

    # --- TAB 2: BIỂU ĐỒ SILHOUETTE ---
    def setup_tab_elbow(self):
        self.fig_frame = tk.Frame(self.tab_elbow, bg="white")
        self.fig_frame.pack(fill="both", expand=True, padx=20, pady=20)
        self.k_options = tk.Frame(self.tab_elbow, bg="#f1f5f9", pady=15)
        self.k_options.pack(fill="x")

    def run_clustering_process(self):
        self.lbl_info.config(text="⏳ Đang tính toán Consensus Matrix...", fg="#f59e0b")
        self.root.update()

        try:
            self.k_values = range(2, 8)  # Tính từ K=2 đến K=7
            self.silhouette_scores = []

            for k in self.k_values:
                # Chạy 3 thuật toán scratch của bạn
                l_gmm = GaussianMixtureModel(k=k).fit_predict(self.X_scaled)
                l_hc = HierarchicalCentroidScratch(k=k).fit_predict(self.X_scaled)
                l_km = kmeansScratch(k=k).fit_predict(self.X_scaled)

                # Chạy Ensemble Consensus
                ens = EnsembleClustering(k=k)
                final_labels = ens.fit_predict(
                    [l_gmm, l_hc, l_km], xScaled=self.X_scaled
                )

                score = silhouette_score(self.X_scaled, final_labels)
                self.silhouette_scores.append(score)

            self.update_elbow_plot()
            self.nb.select(1)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi trong quá trình tính toán: {str(e)}")

    def update_elbow_plot(self):
        for widget in self.fig_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(
            self.k_values,
            self.silhouette_scores,
            marker="o",
            linestyle="-",
            color="#4f46e5",
            linewidth=2,
        )
        ax.set_title(
            "Đánh giá số cụm tối ưu (Silhouette Score)", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Số lượng cụm (K)")
        ax.set_ylabel("Silhouette Score")
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, self.fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        for widget in self.k_options.winfo_children():
            widget.destroy()
        tk.Label(
            self.k_options,
            text="Chọn K để xem phân tích chi tiết:",
            bg="#f1f5f9",
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left", padx=20)

        for i, k in enumerate(self.k_values):
            btn = tk.Button(
                self.k_options,
                text=f"K = {k}",
                command=lambda v=k: self.select_final_k(v),
                bg="white",
                relief="groove",
                width=8,
                cursor="hand2",
            )
            btn.pack(side="left", padx=5)

    # --- TAB 3: PHÂN TÍCH CHI TIẾT CỤM ---
    def setup_tab_analysis(self):
        self.analysis_container = tk.Frame(self.tab_analysis, bg="white")
        self.analysis_container.pack(fill="both", expand=True, padx=20, pady=20)

    def select_final_k(self, k):
        self.selected_k = k
        # Chạy lại một lần cuối với K đã chọn
        l_gmm = GaussianMixtureModel(k=k).fit_predict(self.X_scaled)
        l_hc = HierarchicalCentroidScratch(k=k).fit_predict(self.X_scaled)
        l_km = kmeansScratch(k=k).fit_predict(self.X_scaled)
        ens = EnsembleClustering(k=k)
        self.ensemble_labels = ens.fit_predict(
            [l_gmm, l_hc, l_km], xScaled=self.X_scaled
        )

        self.render_cluster_analysis()
        self.nb.select(2)

    def render_cluster_analysis(self):
        for w in self.analysis_container.winfo_children():
            w.destroy()

        # Scrollbar cho phân tích
        canvas = tk.Canvas(self.analysis_container, bg="white")
        scrollbar = ttk.Scrollbar(
            self.analysis_container, orient="vertical", command=canvas.yview
        )
        scrollable_frame = tk.Frame(canvas, bg="white")

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        colors = ["#10b981", "#f59e0b", "#ef4444", "#3b82f6", "#8b5cf6"]

        for i in range(self.selected_k):
            cluster_df = self.df[self.ensemble_labels == i]
            percent = (len(cluster_df) / len(self.df)) * 100

            card = tk.Frame(
                scrollable_frame,
                bg="white",
                highlightbackground="#e2e8f0",
                highlightthickness=1,
                pady=15,
                padx=15,
            )
            card.pack(fill="x", pady=10, padx=5)

            tk.Label(
                card,
                text=f"NHÓM {i} ({percent:.1f}% bệnh nhân)",
                font=("Segoe UI", 13, "bold"),
                bg="white",
                fg=colors[i % 5],
            ).pack(anchor="w")

            # Hiển thị đặc trưng tiêu biểu
            desc = (
                f"• Thời gian nằm viện TB: {cluster_df['time_in_hospital'].mean():.2f} ngày\n"
                f"• Số loại thuốc TB: {cluster_df['num_medications'].mean():.2f}\n"
                f"• Số lần nhập viện nội trú TB: {cluster_df['number_inpatient'].mean():.2f}"
            )
            tk.Label(
                card, text=desc, font=("Segoe UI", 10), bg="white", justify="left"
            ).pack(anchor="w", pady=5)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    # --- TAB 4: FORM DỰ ĐOÁN ---
    def setup_tab_predict(self):
        self.entries = {}
        fields = [
            ("Thời gian nằm viện (ngày)", "time_in_hospital"),
            ("Số xét nghiệm Lab", "num_lab_procedures"),
            ("Số thủ thuật y tế", "num_procedures"),
            ("Số loại thuốc sử dụng", "num_medications"),
            ("Số lần khám ngoại trú", "number_outpatient"),
            ("Số lần cấp cứu", "number_emergency"),
            ("Số lần nhập viện nội trú", "number_inpatient"),
            ("Số lượng chẩn đoán", "number_diagnoses"),
            ("Chỉ số A1C (0-2)", "A1Cresult"),
            ("Liều Insulin (0-1)", "insulin"),
            ("Thay đổi thuốc (0-1)", "change"),
            ("Sử dụng thuốc tiểu đường (0-1)", "diabetesMed"),
        ]

        main_form = tk.Frame(self.tab_predict, bg="white", pady=30)
        main_form.pack()

        for i, (label, key) in enumerate(fields):
            row, col = i // 2, i % 2
            tk.Label(main_form, text=label, bg="white", font=("Segoe UI", 10)).grid(
                row=row, column=col * 2, padx=15, pady=8, sticky="e"
            )
            ent = tk.Entry(main_form, font=("Segoe UI", 10), width=15)
            ent.grid(row=row, column=col * 2 + 1, pady=8, sticky="w")
            self.entries[key] = ent

        tk.Button(
            self.tab_predict,
            text="🎯 CHẨN ĐOÁN NHÓM NGUY CƠ",
            command=self.predict_new_patient,
            bg="#4f46e5",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            padx=40,
            pady=12,
        ).pack(pady=20)

        self.lbl_res = tk.Label(
            self.tab_predict, text="", font=("Segoe UI", 15, "bold"), bg="white"
        )
        self.lbl_res.pack()

    def predict_new_patient(self):
        if self.ensemble_labels is None:
            messagebox.showwarning(
                "Cảnh báo", "Vui lòng thực hiện phân cụm dữ liệu trước!"
            )
            return

        try:
            # Lấy dữ liệu và scale
            user_input = [float(self.entries[k].get()) for k in self.entries]

            # Tính tâm (centroid) của các cụm hiện tại
            centroids = []
            for i in range(self.selected_k):
                centroids.append(self.X_scaled[self.ensemble_labels == i].mean(axis=0))

            # Scale input người dùng
            # Phải dùng scaler đã fit trên toàn bộ dữ liệu (numeric only)
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            scaler = RobustScaler().fit(self.df[numeric_cols])
            input_scaled = scaler.transform([user_input])

            # Tìm cụm gần nhất bằng Euclidean Distance
            distances = [np.linalg.norm(input_scaled - c) for c in centroids]
            closest_cluster = np.argmin(distances)

            risk_map = {
                0: ("NGUY CƠ THẤP", "#10b981"),
                1: ("NGUY CƠ TRUNG BÌNH", "#f59e0b"),
                2: ("NGUY CƠ CAO", "#ef4444"),
            }
            risk_text, risk_color = risk_map.get(
                closest_cluster % 3
            )  # Demo xoay vòng 3 mức độ

            self.lbl_res.config(
                text=f"KẾT QUẢ: THUỘC NHÓM {closest_cluster} ({risk_text})",
                fg=risk_color,
            )

        except Exception as e:
            messagebox.showerror(
                "Lỗi nhập liệu", "Vui lòng nhập đầy đủ 12 thông số là định dạng số!"
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = DiabetesClusteringApp(root)
    root.mainloop()

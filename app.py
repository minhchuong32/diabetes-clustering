import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
from src.algorithms.gmm_member import GaussianMixtureModel
from src.algorithms.hierarchical_member import HierarchicalCentroidScratch
from src.algorithms.kmeans_member import kmeansScratch
from src.ensemble import EnsembleClustering


class DiabetesClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống Phân cụm Bệnh Tiểu đường")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f4f8")

        # Biến lưu trữ
        self.df = None
        self.X_scaled = None
        self.silhouette_scores = []
        self.k_values = []
        self.selected_k = 3
        self.ensemble_labels = None
        self.cluster_stats = None

        # Style
        self.setup_styles()

        # Tạo giao diện
        self.create_header()
        self.create_notebook()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Button styles
        style.configure(
            "Accent.TButton",
            background="#4f46e5",
            foreground="white",
            font=("Segoe UI", 11, "bold"),
            padding=10,
        )
        style.map("Accent.TButton", background=[("active", "#4338ca")])

        style.configure(
            "Success.TButton",
            background="#10b981",
            foreground="white",
            font=("Segoe UI", 10, "bold"),
            padding=8,
        )

        # Frame styles
        style.configure("Card.TFrame", background="white", relief="flat")

    def create_header(self):
        header_frame = tk.Frame(self.root, bg="#4f46e5", height=100)
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="🏥 Hệ thống Phân cụm Bệnh Tiểu đường",
            font=("Segoe UI", 24, "bold"),
            bg="#4f46e5",
            fg="white",
        )
        title_label.pack(side="left", padx=20, pady=10)

        subtitle_label = tk.Label(
            header_frame,
            text="Ensemble Clustering: GMM + Hierarchical + K-Means",
            font=("Segoe UI", 11),
            bg="#4f46e5",
            fg="#c7d2fe",
        )
        subtitle_label.pack(side="left", padx=20)

    def create_notebook(self):
        # Container
        container = tk.Frame(self.root, bg="#f0f4f8")
        container.pack(fill="both", expand=True, padx=20, pady=10)

        # Notebook
        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill="both", expand=True)

        # Tabs
        self.tab1 = tk.Frame(self.notebook, bg="white")
        self.tab2 = tk.Frame(self.notebook, bg="white")
        self.tab3 = tk.Frame(self.notebook, bg="white")
        self.tab4 = tk.Frame(self.notebook, bg="white")

        self.notebook.add(self.tab1, text="  📁 Bắt đầu  ")
        self.notebook.add(self.tab2, text="  📊 Silhouette  ")
        self.notebook.add(self.tab3, text="  📈 Phân tích cụm  ")
        self.notebook.add(self.tab4, text="  🔮 Dự đoán  ")

        # Tạo nội dung các tab
        self.create_tab1_content()
        self.create_tab2_content()
        self.create_tab3_content()
        self.create_tab4_content()

    def create_tab1_content(self):
        # Card frame
        card = tk.Frame(self.tab1, bg="white", relief="solid", bd=1)
        card.pack(fill="both", expand=True, padx=30, pady=30)

        # Icon và tiêu đề
        icon_label = tk.Label(card, text="⚙️", font=("Segoe UI", 60), bg="white")
        icon_label.pack(pady=(50, 20))

        title = tk.Label(
            card,
            text="Chào mừng đến với hệ thống phân cụm",
            font=("Segoe UI", 18, "bold"),
            bg="white",
            fg="#1f2937",
        )
        title.pack(pady=10)

        desc = tk.Label(
            card,
            text="Tải file CSV và chạy thuật toán Ensemble Clustering\nđể tìm số cụm tối ưu cho dữ liệu của bạn",
            font=("Segoe UI", 12),
            bg="white",
            fg="#6b7280",
            justify="center",
        )
        desc.pack(pady=20)

        # Buttons
        btn_frame = tk.Frame(card, bg="white")
        btn_frame.pack(pady=30)

        load_btn = tk.Button(
            btn_frame,
            text="📂 Tải file CSV",
            command=self.load_csv,
            bg="#6366f1",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            padx=30,
            pady=15,
            relief="flat",
            cursor="hand2",
        )
        load_btn.pack(side="left", padx=10)

        self.run_btn = tk.Button(
            btn_frame,
            text="▶️ Chạy Ensemble Clustering",
            command=self.run_ensemble,
            bg="#10b981",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            padx=30,
            pady=15,
            relief="flat",
            cursor="hand2",
            state="disabled",
        )
        self.run_btn.pack(side="left", padx=10)

        # Status
        self.status_label = tk.Label(
            card,
            text="Chưa tải dữ liệu",
            font=("Segoe UI", 11),
            bg="white",
            fg="#ef4444",
        )
        self.status_label.pack(pady=20)

    def create_tab2_content(self):
        # Container chính
        container = tk.Frame(self.tab2, bg="white")
        container.pack(fill="both", expand=True, padx=20, pady=20)

        # Tiêu đề và mô tả
        title = tk.Label(
            container,
            text="📊 Biểu đồ Silhouette Score",
            font=("Segoe UI", 16, "bold"),
            bg="white",
            fg="#1f2937",
        )
        title.pack(pady=(0, 10))

        desc = tk.Label(
            container,
            text="Chọn số cụm (k) phía dưới để xem phân tích chi tiết",
            font=("Segoe UI", 10),
            bg="white",
            fg="#6b7280",
        )
        desc.pack(pady=(0, 10))

        # Frame chứa biểu đồ - KHÔNG dùng expand=True
        self.chart_frame = tk.Frame(container, bg="white", height=400)
        self.chart_frame.pack(fill="both", pady=10)
        self.chart_frame.pack_propagate(False)  # Giữ chiều cao cố định

        # Frame chứa các nút chọn K - Đặt ở dưới cùng
        self.k_select_frame = tk.Frame(container, bg="#f8fafc", relief="solid", bd=1)
        self.k_select_frame.pack(fill="x", side="bottom", pady=10)

    def create_tab3_content(self):
        # Container với scrollbar
        container = tk.Frame(self.tab3, bg="white")
        container.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        title = tk.Label(
            container,
            text="📈 Phân tích chi tiết các cụm",
            font=("Segoe UI", 16, "bold"),
            bg="white",
            fg="#1f2937",
        )
        title.pack(pady=(0, 20))

        # Canvas với scrollbar
        canvas = tk.Canvas(container, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.cluster_content = tk.Frame(canvas, bg="white")

        self.cluster_content.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.cluster_content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_tab4_content(self):
        # Container chính với scrollbar
        main_container = tk.Frame(self.tab4, bg="white")
        main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # Canvas với scrollbar
        canvas = tk.Canvas(main_container, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            main_container, orient="vertical", command=canvas.yview
        )
        container = tk.Frame(canvas, bg="white")

        container.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Title
        title = tk.Label(
            container,
            text="🔮 Nhập dữ liệu để dự đoán cụm",
            font=("Segoe UI", 16, "bold"),
            bg="white",
            fg="#1f2937",
        )
        title.pack(pady=(20, 30), padx=40)

        # Input frame
        input_frame = tk.Frame(container, bg="white")
        input_frame.pack(fill="x", padx=40)

        # Các trường nhập liệu với placeholder - ĐẦY ĐỦ 12 CỘT
        fields = [
            ("Thời gian nằm viện (ngày):", "time_in_hospital", "VD: 3"),
            ("Số thủ thuật xét nghiệm:", "num_lab_procedures", "VD: 45"),
            ("Số thủ thuật y tế:", "num_procedures", "VD: 2"),
            ("Số loại thuốc:", "num_medications", "VD: 15"),
            ("Số lần khám ngoại trú:", "number_outpatient", "VD: 0"),
            ("Số lần cấp cứu:", "number_emergency", "VD: 1"),
            ("Số lần nội trú:", "number_inpatient", "VD: 0"),
            ("Số chẩn đoán:", "number_diagnoses", "VD: 7"),
            ("Kết quả A1C (0/1/2):", "A1Cresult", "VD: 0.0"),
            ("Dùng insulin (0/1):", "insulin", "VD: 1"),
            ("Thay đổi thuốc (0/1):", "change", "VD: 0"),
            ("Dùng thuốc tiểu đường (0/1):", "diabetesMed", "VD: 1"),
        ]

        self.input_entries = {}

        for i, (label_text, field_name, placeholder) in enumerate(fields):
            row = i // 2
            col = i % 2

            field_frame = tk.Frame(input_frame, bg="white")
            field_frame.grid(row=row, column=col, padx=20, pady=10, sticky="ew")

            label = tk.Label(
                field_frame,
                text=label_text,
                font=("Segoe UI", 10, "bold"),
                bg="white",
                fg="#374151",
            )
            label.pack(anchor="w")

            entry = tk.Entry(
                field_frame,
                font=("Segoe UI", 10),
                relief="solid",
                bd=1,
                width=25,
                fg="#9ca3af",
            )
            entry.pack(fill="x", pady=(5, 0))
            entry.insert(0, placeholder)

            # Xử lý placeholder
            def on_focus_in(event, e=entry, ph=placeholder):
                if e.get() == ph:
                    e.delete(0, tk.END)
                    e.config(fg="#1f2937")

            def on_focus_out(event, e=entry, ph=placeholder):
                if e.get() == "":
                    e.insert(0, ph)
                    e.config(fg="#9ca3af")

            entry.bind("<FocusIn>", on_focus_in)
            entry.bind("<FocusOut>", on_focus_out)

            self.input_entries[field_name] = entry

        input_frame.columnconfigure(0, weight=1)
        input_frame.columnconfigure(1, weight=1)

        # Predict button
        predict_btn = tk.Button(
            container,
            text="🎯 Dự đoán cụm",
            command=self.predict_cluster,
            bg="#8b5cf6",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            padx=40,
            pady=15,
            relief="flat",
            cursor="hand2",
        )
        predict_btn.pack(pady=30)

        # Result frame
        self.result_frame = tk.Frame(container, bg="#f0fdf4", relief="solid", bd=2)
        self.result_frame.pack(fill="x", pady=20, padx=40)
        self.result_frame.pack_forget()  # Ẩn ban đầu

    def predict_cluster(self):
        if self.ensemble_labels is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chạy phân tích cụm trước!")
            return

        # Lấy dữ liệu từ form THEO ĐÚNG THỨ TỰ CỘT TRONG CSV
        try:
            input_data = []

            # Thứ tự cột trong CSV
            column_order = [
                "time_in_hospital",
                "num_lab_procedures",
                "num_procedures",
                "num_medications",
                "number_outpatient",
                "number_emergency",
                "number_inpatient",
                "number_diagnoses",
                "A1Cresult",
                "insulin",
                "change",
                "diabetesMed",
            ]

            for col_name in column_order:
                if col_name not in self.input_entries:
                    messagebox.showerror("Lỗi", f"Thiếu trường: {col_name}")
                    return

                val = self.input_entries[col_name].get().strip()

                # Kiểm tra nếu là placeholder hoặc trống
                if val.startswith("VD:") or val == "":
                    messagebox.showerror(
                        "Lỗi", f"Vui lòng nhập giá trị cho trường: {col_name}"
                    )
                    return

                value = float(val)
                input_data.append(value)

            # Kiểm tra số lượng
            if len(input_data) != 12:
                messagebox.showerror(
                    "Lỗi", f"Cần đúng 12 giá trị, nhận được {len(input_data)}"
                )
                return

            input_array = np.array([input_data])

            # Chuẩn hóa
            scaler = RobustScaler()
            scaler.fit(self.df.values)
            input_scaled = scaler.transform(input_array)

            # Tính khoảng cách đến các tâm cụm
            cluster_centers = []
            for i in range(self.selected_k):
                cluster_data = self.X_scaled[self.ensemble_labels == i]
                center = cluster_data.mean(axis=0)
                cluster_centers.append(center)

            distances = [
                np.linalg.norm(input_scaled - center) for center in cluster_centers
            ]
            predicted_cluster = np.argmin(distances)

            # Hiển thị kết quả
            self.show_prediction_result(predicted_cluster)

        except ValueError as e:
            messagebox.showerror(
                "Lỗi",
                f"Vui lòng nhập số hợp lệ cho tất cả các trường!\nChi tiết: {str(e)}",
            )

    def load_csv(self):
        file_path = filedialog.askopenfilename(
            title="Chọn file CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )

        if file_path:
            try:
                self.df = pd.read_csv(file_path)

                # Chuẩn hóa dữ liệu
                scaler = RobustScaler()
                self.X_scaled = scaler.fit_transform(self.df.values)

                self.status_label.config(
                    text=f"✅ Đã tải {len(self.df)} dòng dữ liệu", fg="#10b981"
                )
                self.run_btn.config(state="normal")

                messagebox.showinfo(
                    "Thành công", f"Đã tải file với {len(self.df)} dòng dữ liệu"
                )
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể đọc file: {str(e)}")

    def run_ensemble(self):
        if self.X_scaled is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng tải file CSV trước!")
            return

        self.status_label.config(
            text="⏳ Đang chạy Ensemble Clustering...", fg="#f59e0b"
        )
        self.root.update()

        try:
            # Chạy ensemble cho các k khác nhau
            k_range = range(2, 11)
            self.k_values = list(k_range)
            self.silhouette_scores = []

            for k in k_range:
                # Chạy 3 thuật toán
                gmm = GaussianMixtureModel(k=k)
                labels_gmm = gmm.fit_predict(self.X_scaled)

                hc = HierarchicalCentroidScratch(k=k)
                labels_hc = hc.fit_predict(self.X_scaled)

                km = kmeansScratch(k=k)
                labels_km = km.fit_predict(self.X_scaled)

                # Ensemble
                ensemble = EnsembleClustering(k=k)
                final_labels = ensemble.fit_predict([labels_gmm, labels_hc, labels_km])

                # Tính Silhouette
                score = silhouette_score(self.X_scaled, final_labels)
                self.silhouette_scores.append(score)

                print(f"K={k}: Silhouette Score = {score:.4f}")

            # Chuyển sang tab Silhouette
            self.plot_silhouette()
            self.notebook.select(1)

            self.status_label.config(text="✅ Hoàn thành!", fg="#10b981")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi chạy ensemble: {str(e)}")
            self.status_label.config(text="❌ Lỗi xảy ra", fg="#ef4444")

    def plot_silhouette(self):
        # Xóa biểu đồ cũ
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        # Vẽ biểu đồ mới
        fig = Figure(figsize=(10, 5), dpi=100)
        ax = fig.add_subplot(111)

        ax.plot(
            self.k_values,
            self.silhouette_scores,
            marker="o",
            linewidth=2,
            markersize=8,
            color="#4f46e5",
        )
        ax.set_xlabel("Số cụm (K)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Silhouette Score", fontsize=12, fontweight="bold")
        ax.set_title("Đánh giá số cụm tối ưu", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(self.k_values)

        # Đánh dấu điểm cao nhất
        max_idx = np.argmax(self.silhouette_scores)
        ax.scatter(
            [self.k_values[max_idx]],
            [self.silhouette_scores[max_idx]],
            color="#10b981",
            s=200,
            zorder=5,
            edgecolors="white",
            linewidth=2,
        )

        canvas = FigureCanvasTkAgg(fig, self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # Tạo các nút chọn K
        for widget in self.k_select_frame.winfo_children():
            widget.destroy()

        # Tiêu đề nhỏ
        tk.Label(
            self.k_select_frame,
            text="CHỌN SỐ CỤM ĐỂ XEM PHÂN TÍCH:",
            font=("Segoe UI", 10, "bold"),
            bg="#f8fafc",
        ).pack(pady=5)

        # Dùng một frame phụ để căn giữa các nút
        btn_container = tk.Frame(self.k_select_frame, bg="#f8fafc")
        btn_container.pack()

        max_idx = np.argmax(self.silhouette_scores)
        for i, k in enumerate(self.k_values):
            is_best = i == max_idx
            # Tạo nút
            btn = tk.Button(
                btn_container,  # Đặt vào container mới
                text=f"K = {k}\n({self.silhouette_scores[i]:.3f})",
                command=lambda val=k: self.select_k(val),
                bg="#10b981" if is_best else "#e2e8f0",
                fg="white" if is_best else "#475569",
                font=("Segoe UI", 9, "bold"),
                width=10,
                padx=5,
                pady=5,
                relief="flat",
                cursor="hand2",
            )
            btn.pack(side="left", padx=10, pady=5)

        self.root.update()

    def select_k(self, k):
        self.selected_k = k

        # Chạy ensemble với k được chọn
        gmm = GaussianMixtureModel(k=k)
        labels_gmm = gmm.fit_predict(self.X_scaled)

        hc = HierarchicalCentroidScratch(k=k)
        labels_hc = hc.fit_predict(self.X_scaled)

        km = kmeansScratch(k=k)
        labels_km = km.fit_predict(self.X_scaled)

        ensemble = EnsembleClustering(k=k)
        self.ensemble_labels = ensemble.fit_predict([labels_gmm, labels_hc, labels_km])

        # Phân tích các cụm
        self.analyze_clusters()
        self.notebook.select(2)

    def analyze_clusters(self):
        # Xóa nội dung cũ
        for widget in self.cluster_content.winfo_children():
            widget.destroy()

        # Tính thống kê cho mỗi cụm
        self.cluster_stats = []
        colors = ["#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899"]
        risk_levels = ["Nguy cơ thấp", "Nguy cơ trung bình", "Nguy cơ cao"]

        for i in range(self.selected_k):
            cluster_data = self.df[self.ensemble_labels == i]

            stats = {
                "cluster_id": i,
                "name": risk_levels[i % 3],
                "size": len(cluster_data),
                "percentage": (len(cluster_data) / len(self.df)) * 100,
                "means": cluster_data.mean().to_dict(),
            }
            self.cluster_stats.append(stats)

            # Tạo card cho mỗi cụm
            card = tk.Frame(
                self.cluster_content, bg=colors[i % len(colors)], relief="solid", bd=0
            )
            card.pack(fill="x", pady=10, padx=20)

            # Header
            header = tk.Frame(card, bg=colors[i % len(colors)])
            header.pack(fill="x", padx=20, pady=15)

            title_label = tk.Label(
                header,
                text=f"Cụm {i}",
                font=("Segoe UI", 16, "bold"),
                bg=colors[i % len(colors)],
                fg="white",
            )
            title_label.pack(side="left")

            size_label = tk.Label(
                header,
                text=f"{stats['percentage']:.1f}% ({stats['size']} bệnh nhân)",
                font=("Segoe UI", 11, "bold"),
                bg="white",
                fg=colors[i % len(colors)],
                padx=15,
                pady=5,
            )
            size_label.pack(side="right")

            # Risk level
            risk_label = tk.Label(
                card,
                text=stats["name"],
                font=("Segoe UI", 14, "bold"),
                bg=colors[i % len(colors)],
                fg="white",
            )
            risk_label.pack(padx=20, pady=(0, 10))

            # Stats content
            stats_frame = tk.Frame(card, bg="white")
            stats_frame.pack(fill="x", padx=20, pady=(0, 20))

            col_names = list(self.df.columns)
            for j, col in enumerate(col_names[:4]):  # Hiển thị 4 cột đầu
                stat_row = tk.Frame(stats_frame, bg="white")
                stat_row.pack(fill="x", pady=5, padx=15)

                label = tk.Label(
                    stat_row,
                    text=f"{col}:",
                    font=("Segoe UI", 10),
                    bg="white",
                    fg="#6b7280",
                )
                label.pack(side="left")

                value = tk.Label(
                    stat_row,
                    text=f"{stats['means'][col]:.2f}",
                    font=("Segoe UI", 10, "bold"),
                    bg="white",
                    fg="#1f2937",
                )
                value.pack(side="right")

    def show_prediction_result(self, cluster_id):
        # Xóa kết quả cũ
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        self.result_frame.pack(fill="x", pady=20)

        colors = ["#10b981", "#f59e0b", "#ef4444"]
        risk_levels = ["Nguy cơ thấp", "Nguy cơ trung bình", "Nguy cơ cao"]

        # Title
        title = tk.Label(
            self.result_frame,
            text="🎯 Kết quả dự đoán",
            font=("Segoe UI", 16, "bold"),
            bg="#f0fdf4",
            fg="#1f2937",
        )
        title.pack(pady=15)

        # Cluster info
        info_frame = tk.Frame(self.result_frame, bg="#f0fdf4")
        info_frame.pack(pady=10)

        cluster_label = tk.Label(
            info_frame,
            text=f"Thuộc cụm:",
            font=("Segoe UI", 12),
            bg="#f0fdf4",
            fg="#6b7280",
        )
        cluster_label.grid(row=0, column=0, padx=10, sticky="e")

        cluster_value = tk.Label(
            info_frame,
            text=f"Cụm {cluster_id}",
            font=("Segoe UI", 14, "bold"),
            bg="#f0fdf4",
            fg=colors[cluster_id % 3],
        )
        cluster_value.grid(row=0, column=1, padx=10, sticky="w")

        risk_label = tk.Label(
            info_frame,
            text=f"Phân loại:",
            font=("Segoe UI", 12),
            bg="#f0fdf4",
            fg="#6b7280",
        )
        risk_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")

        risk_value = tk.Label(
            info_frame,
            text=risk_levels[cluster_id % 3],
            font=("Segoe UI", 14, "bold"),
            bg="#f0fdf4",
            fg=colors[cluster_id % 3],
        )
        risk_value.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        # Recommendation
        recommendations = {
            0: "Chỉ số sức khỏe của bạn tương đối tốt. Hãy duy trì lối sống lành mạnh, ăn uống điều độ và vận động thường xuyên.",
            1: "Bạn nên chú ý theo dõi sức khỏe và điều chỉnh chế độ ăn uống. Nên tham khảo ý kiến bác sĩ để có kế hoạch phòng ngừa phù hợp.",
            2: "Bạn có nguy cơ cao về bệnh tiểu đường. Khuyến nghị gặp bác sĩ chuyên khoa để được tư vấn và theo dõi sát sao.",
        }

        rec_frame = tk.Frame(self.result_frame, bg="white", relief="solid", bd=1)
        rec_frame.pack(fill="x", padx=20, pady=15)

        rec_title = tk.Label(
            rec_frame,
            text="💡 Khuyến nghị:",
            font=("Segoe UI", 11, "bold"),
            bg="white",
            fg="#1f2937",
        )
        rec_title.pack(anchor="w", padx=15, pady=(10, 5))

        rec_text = tk.Label(
            rec_frame,
            text=recommendations[cluster_id % 3],
            font=("Segoe UI", 10),
            bg="white",
            fg="#4b5563",
            wraplength=700,
            justify="left",
        )
        rec_text.pack(anchor="w", padx=15, pady=(5, 15))


# Chạy ứng dụng
if __name__ == "__main__":
    root = tk.Tk()
    app = DiabetesClusteringApp(root)
    root.mainloop()

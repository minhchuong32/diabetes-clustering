import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import sys
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.preprocessing import RobustScaler

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.ensemble import ensemble_lib, ensemble_lib_single
from src.analysis import analysis_lib


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
                self.file_path = file_path
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
        self.plot_container = tk.Frame(self.tab_elbow, bg="white")
        self.plot_container.pack(fill="both", expand=True, padx=20, pady=20)
        self.k_options = tk.Frame(self.tab_elbow, bg="#f1f5f9", pady=15)
        self.k_options.pack(fill="x")

    def run_clustering_process(self):
        self.lbl_info.config(text="⏳ Đang tính toán Consensus Matrix...", fg="#f59e0b")
        self.root.update()

        try:
            k_range = range(2, 11)
            results = ensemble_lib(self.X_scaled, kRange=k_range)

            for w in self.plot_container.winfo_children(): w.destroy()

            fig = Figure(figsize=(10, 5), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(k_range, results['ensemble'], 'o-', color='red', label='Ensemble', linewidth=3)
            ax.plot(k_range, results['gmm'], 's--', alpha=0.5, label='GMM')
            ax.plot(k_range, results['km'], 'x--', alpha=0.5, label='K-Means')
            ax.plot(k_range, results['hier'], '^--', alpha=0.5, label='Hierachical')

            ax.set_title("So sánh Silhouette Score (Library Models)")
            ax.set_xlabel("Số cụm K")
            ax.set_ylabel("Silhouette Score")
            ax.legend()
            ax.grid(True, linestyle=':')
            canvas = FigureCanvasTkAgg(fig, self.plot_container)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            self.selected_k = 2
            self.ensemble_labels = ensemble_lib_single(self.X_scaled, k=self.selected_k)
            self.nb.select(1)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi trong quá trình tính toán: {str(e)}")

    # --- TAB 3: PHÂN TÍCH CHI TIẾT CỤM ---
    def setup_tab_analysis(self):
        top_bar = tk.Frame(self.tab_analysis, bg="#f8fafc", pady=10)
        top_bar.pack(fill="x")

        tk.Label(top_bar, text="Chọn K tối ưu:", bg="#f8fafc").pack(side="left", padx=10)
        self.spin_k = tk.Spinbox(top_bar, from_=2, to=10, width=5)
        self.spin_k.pack(side="left", padx=5)

        tk.Button(top_bar, text="XUẤT BẢNG THỐNG KÊ PHÂN CỤM BỆNH TIỂU ĐƯỜNG", command=self.run_full_analysis,
                  bg="#10b981", fg="white", font=("Segoe UI", 9, "bold")).pack(side="left", padx=20)

        # Bảng hiển thị
        self.tree_frame = tk.Frame(self.tab_analysis)
        self.tree_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def run_full_analysis(self):
        if self.file_path is None:
            messagebox.showwarning("Lỗi", "Vui lòng tải dữ liệu trước!")
            return

        k = int(self.spin_k.get())
        stats_df = analysis_lib(self.file_path, k=k)

        self.display_stats_in_tree(stats_df)

    def display_stats_in_tree(self, df):
        for w in self.tree_frame.winfo_children(): w.destroy()
        
        displayDf = df.reset_index()
        columnNames = list(displayDf.columns)
        
        tree = ttk.Treeview(self.tree_frame, columns=columnNames, show="headings")

        for col in columnNames:
            tree.heading(col, text=col.upper())
            tree.column(col, width=120, anchor="center")

        for _, row in displayDf.iterrows():
            cleanVals = [round(v, 4) if isinstance(v, (float, np.float64)) else v for v in row]
            tree.insert("", "end", values=cleanVals)

        vsb = ttk.Scrollbar(self.tree_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(self.tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        self.tree_frame.grid_columnconfigure(0, weight=1)
        self.tree_frame.grid_rowconfigure(0, weight=1)


    # --- TAB 4: FORM DỰ ĐOÁN ---
    def setup_tab_predict(self):
        self.entries = {}
        fields = [
            ("Chỉ số A1C (0-8)", "A1Cresult"),
            ("Số thủ thuật y tế", "num_procedures"),
            ("Thay đổi thuốc (0-1)", "change"),
            ("Số lượng chẩn đoán", "number_diagnoses"),
            ("Sử dụng thuốc tiểu đường (0-1)", "diabetesMed"),
            ("Số lần cấp cứu", "number_emergency"),
            ("Liều Insulin (0-2)", "insulin"),
            ("Số lần nhập viện nội trú", "number_inpatient"),
            ("Số xét nghiệm Lab", "num_lab_procedures"),
            ("Số lần khám ngoại trú", "number_outpatient"),
            ("Số loại thuốc sử dụng", "num_medications"),
            ("Thời gian nằm viện (ngày)", "time_in_hospital")
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
            user_input = [float(self.entries[k].get()) for k in self.entries]

            #tâm
            centroids = []
            for i in range(self.selected_k):
                centroids.append(self.X_scaled[self.ensemble_labels == i].mean(axis=0))

            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            scaler = RobustScaler().fit(self.df[numeric_cols].values)
            input_scaled = scaler.transform([user_input])

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

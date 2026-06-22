import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import pandas as pd
import csv
from scipy.stats import pearsonr
import time
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Sample_preprocessing_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Outlier removal and PLS analysis system")
        self.root.geometry("1000x800")

        self.original_data = None
        self.y_data = None
        self.x_data = None
        self.dependent_vars = 1
        self.polynomial_degree = 1
        self.pls_running = None

        self.do_smoothing = tk.BooleanVar(value=True)
        self.do_derivative = tk.BooleanVar(value=True)

        self.use_cars = tk.BooleanVar(value=False)

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=0, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        self.load_btn = ttk.Button(button_frame, text="Load data file", command=self.load_data)
        self.load_btn.grid(row=0, column=0, padx=5)

        self.analyze_btn = ttk.Button(button_frame, text="Spectral preprocessing", command=self.pca_analysis)
        self.analyze_btn.grid(row=0, column=1, padx=5)

        self.pls_btn = ttk.Button(button_frame, text="PLS", command=self.pls_analysis)
        self.pls_btn.grid(row=0, column=2, padx=5)

        self.export_btn = ttk.Button(button_frame, text="Export data", command=self.export_data)
        self.export_btn.grid(row=0, column=3, padx=5)

        self.clear_btn = ttk.Button(button_frame, text="Clear", command=self.clear_all)
        self.clear_btn.grid(row=0, column=4, padx=5)

        pls_frame = ttk.LabelFrame(main_frame, text="Parameter settings", padding="5")
        pls_frame.grid(row=1, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))

        ttk.Label(pls_frame, text="Number of dependent variables:").grid(row=0, column=0, padx=5)
        self.dependent_var = tk.StringVar(value="1")
        self.dependent_entry = ttk.Entry(pls_frame, textvariable=self.dependent_var, width=10)
        self.dependent_entry.grid(row=0, column=1, padx=5)

        ttk.Label(pls_frame, text="Anomaly detection threshold:").grid(row=0, column=2, padx=5)
        self.threshold_var = tk.StringVar(value="10.0")
        self.threshold_entry = ttk.Entry(pls_frame, textvariable=self.threshold_var, width=10)
        self.threshold_entry.grid(row=0, column=3, padx=5)

        ttk.Label(pls_frame, text="Polynomial degree:").grid(row=0, column=4, padx=5)
        self.polynomial_var = tk.StringVar(value="1")
        self.polynomial_entry = ttk.Entry(pls_frame, textvariable=self.polynomial_var, width=10)
        self.polynomial_entry.grid(row=0, column=5, padx=5)

        ttk.Label(pls_frame, text="Training set ratio:").grid(row=0, column=6, padx=5)
        self.train_ratio_var = tk.StringVar(value="0.7")
        self.train_ratio_entry = ttk.Entry(pls_frame, textvariable=self.train_ratio_var, width=10)
        self.train_ratio_entry.grid(row=0, column=7, padx=5)

        ttk.Label(pls_frame, text="Preprocessing:").grid(row=1, column=0, padx=(20, 5))

        ttk.Checkbutton(pls_frame, text="Three-point smoothing", variable=self.do_smoothing).grid(row=1, column=1, padx=5)
        ttk.Checkbutton(pls_frame, text="First derivative", variable=self.do_derivative).grid(row=1, column=2, padx=5)
        ttk.Checkbutton(pls_frame, text="CARS", variable=self.use_cars).grid(row=1, column=3, padx=10)

        listbox_frame = ttk.Frame(main_frame)
        listbox_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))

        scrollbar_y = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL)
        scrollbar_x = ttk.Scrollbar(listbox_frame, orient=tk.HORIZONTAL)

        self.listbox = tk.Listbox(listbox_frame,
                                  yscrollcommand=scrollbar_y.set,
                                  xscrollcommand=scrollbar_x.set,
                                  width=80, height=20)

        scrollbar_y.config(command=self.listbox.yview)
        scrollbar_x.config(command=self.listbox.xview)

        self.listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        scrollbar_x.grid(row=1, column=0, sticky=(tk.W, tk.E))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        listbox_frame.columnconfigure(0, weight=1)
        listbox_frame.rowconfigure(0, weight=1)

    def start_pls_progress(self):
        self.pls_start_time = time.time()
        self.pls_running = True
        self._update_progress_loop()

    def _update_progress_loop(self):
        if not getattr(self, 'pls_running', False):
            return

        elapsed = time.time() - self.pls_start_time
        self.update_title(f"PLS...Time: {elapsed:.1f}s")

        self.root.after(500, self._update_progress_loop)

    def update_title(self, message):
        self.root.title(f"Outlier removal and PLS analysis system - {message}")

    def load_data(self):
        file_path = filedialog.askopenfilename(
            title="Select data file",
            filetypes=[("TXT", "*.txt"),
            ("CSV", "*.csv"),
            ("Excel", "*.xlsx"),
            ("ALL", "*.*")]
        )

        if file_path:
            try:
                if file_path.endswith('.txt'):
                    data = np.loadtxt(file_path)
                elif file_path.endswith('.csv'):
                    data = np.genfromtxt(file_path, delimiter=',')
                elif file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                    data = df.values

                self.original_data = data
                self.y_data = data
                self.x_data = data

                self.update_listbox(data)
                messagebox.showinfo("Success", f"Data loaded successfully\nData shape: {data.shape}")
                self.update_title("Data loaded")

            except Exception as e:
                messagebox.showerror("Error", f"Error loading data: {str(e)}")

    def export_data(self):
        if self.y_data is None:
            messagebox.showwarning("Warning", "No data to export")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                title="Export data",
                defaultextension=".txt",
                filetypes=[
                    ("TXT", "*.txt"),
                    ("CSV", "*.csv"),
                    ("Excel", "*.xlsx"),
                    ("ALL", "*.*")
                ]
            )

            if file_path:
                if file_path.endswith('.txt'):
                    np.savetxt(file_path, self.y_data, fmt='%.6f', delimiter='\t')
                elif file_path.endswith('.csv'):
                    np.savetxt(file_path, self.y_data, fmt='%.6f', delimiter=',')
                elif file_path.endswith('.xlsx'):
                    df = pd.DataFrame(self.y_data)
                    df.to_excel(file_path, index=False, header=False)

                messagebox.showinfo("Success", f"Data successfully exported to:\n{file_path}")
                self.update_title("Data exported")

        except Exception as e:
            messagebox.showerror("Error", f"Error exporting data: {str(e)}")

    def transpose_data(self):
        if self.y_data is not None:
            self.y_data = self.y_data.T
            self.update_listbox(self.y_data)
        else:
            messagebox.showwarning("Warning", "Please load data first")

    def pca_analysis(self):
        if self.original_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return

        try:
            self.update_title("Principal component analysis in progress...")
            self.root.update()

            self.dependent_vars = int(self.dependent_var.get())

            data_T = self.original_data.T
            self.y_data = data_T[:, :-self.dependent_vars].astype(float)
            self.x_data = data_T[:, -self.dependent_vars:].astype(float)

            threshold = float(self.threshold_var.get())

            x = self.y_data.astype(float)

            n = x.shape[0]

            scaler = StandardScaler()
            sr = scaler.fit_transform(self.y_data)

            pca = PCA()
            pca.fit(sr)

            pcs = pca.components_.T
            newdata = pca.transform(sr)

            variances = pca.explained_variance_

            g = variances / np.sum(variances)

            d = np.zeros(n)

            n_components = min(3, pcs.shape[1])

            for i in range(n):
                score = 0
                for j in range(n_components):
                    projection = np.dot(x[i, :], pcs[:, j])
                    score += g[j] * projection
                d[i] = score

            upper_threshold = np.mean(d) + threshold
            lower_threshold = np.mean(d) - threshold
            outliers = np.where((d > upper_threshold) | (d < lower_threshold))[0]

            self.create_outlier_plot(d, threshold)

            if len(outliers) > 0:
                self.y_data = x
                self.y_data = np.column_stack([self.y_data, self.x_data])
                cleaned_data = np.delete(self.y_data, outliers, axis=0)

                self.y_data = cleaned_data

                outlier_text = f"Detected {len(outliers)} outlier samples:\n" + ", ".join([f"Sample {i + 1}" for i in outliers])
                result_text = f"{outlier_text}\n\nThese outlier samples have been automatically removed from the data.\nData reduced from {n} samples to {len(cleaned_data)} samples."
                messagebox.showinfo("Outlier Detection and Removal", result_text)
            else:
                self.y_data = x
                self.y_data = np.column_stack([self.y_data, self.x_data])

                messagebox.showinfo("Outlier detection", "No outlier samples detected. Data remains unchanged.")

            self.transpose_data()
            self.x_data = self.y_data[-self.dependent_vars:, :]
            self.y_data = self.y_data[:-self.dependent_vars, :]
            self.smooth_derivative_combined()

            self.y_data = np.vstack([self.y_data, self.x_data])
            self.update_listbox(self.y_data)

            self.update_title("Principal component analysis completed")

        except Exception as e:
            messagebox.showerror("Error", f"Error during principal component analysis: {str(e)}")

    def create_outlier_plot(self, d, threshold):
        fig, ax = plt.subplots(figsize=(12, 6))

        n = len(d)
        indices = np.arange(1, n + 1)
        mean_d = np.mean(d)

        ax.plot(indices, d, '*r', markersize=8, label='Principal component scores', alpha=0.7)

        x_range = np.linspace(0.5, n + 0.5, 100)
        ax.plot(x_range, np.full(100, mean_d + threshold), 'r-',
                linewidth=2, label='Upper threshold')
        ax.plot(x_range, np.full(100, mean_d - threshold), 'r-',
                linewidth=2, label='Lower threshold')
        ax.axhline(y=mean_d, color='b', linestyle='--', alpha=0.7, label='Average')

        outliers = np.where((d > mean_d + threshold) | (d < mean_d - threshold))[0]
        if len(outliers) > 0:
            ax.plot(indices[outliers], d[outliers], 'ko', markersize=10,
                    fillstyle='none', markeredgewidth=2, label='Outlier')

            for i in outliers:
                ax.annotate(f'{i + 1}', (indices[i], d[i]),
                            xytext=(5, 5), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        ax.set_xlabel('Sample index', fontsize=12)
        ax.set_ylabel('PCA scores', fontsize=12)
        ax.set_title('Outlier Removal – PCA Score Analysis', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, n + 0.5)

        plt.tight_layout()
        plt.show(block=False)

    def clear_all(self):
        self.y_data = None
        self.listbox.delete(0, tk.END)
        self.update_title("Data cleared")

    def update_listbox(self, data):
        self.listbox.delete(0, tk.END)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        for i in range(data.shape[0]):
            row_str = "  ".join([f"{val:.6f}" for val in data[i, :]])
            self.listbox.insert(tk.END, row_str)

    def first_derivative(self, data):
        return np.diff(data, axis=0)

    def three_point_smoothing(self, data):
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        n_rows, n_cols = data.shape
        smoothed = np.zeros_like(data)

        for col in range(n_cols):
            smoothed[0, col] = data[0, col]
            smoothed[-1, col] = data[-1, col]

            for i in range(1, n_rows - 1):
                smoothed[i, col] = (data[i - 1, col] + data[i, col] + data[i + 1, col]) / 3.0

        return smoothed

    def smooth_derivative_combined(self):
        if self.y_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return

        try:
            data = self.y_data.copy()

            if self.do_smoothing.get():
                data = self.three_point_smoothing(data)

            if self.do_derivative.get():
                if data.shape[0] < 2:
                    messagebox.showwarning("Warning", "Insufficient data rows to calculate the first derivative")
                    return
                data = self.first_derivative(data)

            self.y_data = data
            self.update_listbox(data)
        except Exception as e:
            messagebox.showerror("Error", f"Error during smoothing and derivative processing: {str(e)}")

    def cars_variable_selection(self, X, y, N=50, f=0.1, cv_folds=5, max_components=15):
        if y.ndim == 2:
            y = y[:, 0] if y.shape[1] > 1 else y.ravel()

        n_samples, n_vars = X.shape
        best_rmsecv = np.inf
        best_subset = None
        rmsecv_history = []

        print(f"Starting CARS variable selection (N={N}, samples={n_samples}, variables={n_vars})...")

        prev_kept = None

        for i in range(N):
            ratio = np.exp(np.log(2.0 / n_vars) * i / (N - 1))
            n_keep = max(2, int(n_vars * ratio + 0.5))

            if i == 0:
                kept_idx = np.random.choice(n_vars, size=n_keep, replace=False)
            else:
                n_comp_temp = min(max_components, len(prev_kept), n_samples - 2)
                n_comp_temp = max(1, n_comp_temp)
                pls_temp = PLSRegression(n_components=n_comp_temp)
                pls_temp.fit(X[:, prev_kept], y)
                coef = np.abs(pls_temp.coef_.ravel())

                weights = np.zeros(n_vars)
                weights[prev_kept] = coef
                weights += 1e-8
                weights /= weights.sum()

                kept_idx = np.random.choice(n_vars, size=n_keep, replace=False, p=weights)

            prev_kept = kept_idx.copy()

            rmse_list = []
            cv = LeaveOneOut() if n_samples < 12 else KFold(n_splits=min(cv_folds, n_samples), shuffle=True,
                                                            random_state=i)

            for train_idx, val_idx in cv.split(range(n_samples)):
                if len(val_idx) == 0 or len(train_idx) < 4:
                    continue
                n_comp = min(max_components, len(kept_idx), len(train_idx) - 3)
                if n_comp < 1:
                    continue

                try:
                    pls = PLSRegression(n_components=n_comp)
                    pls.fit(X[train_idx][:, kept_idx], y[train_idx])
                    pred = pls.predict(X[val_idx][:, kept_idx]).ravel()
                    rmse = np.sqrt(np.mean((pred - y[val_idx]) ** 2))
                    rmse_list.append(rmse)
                except:
                    continue

            rmsecv = np.mean(rmse_list) if rmse_list else 999
            rmsecv_history.append(rmsecv)

            if rmsecv < best_rmsecv:
                best_rmsecv = rmsecv
                best_subset = kept_idx.copy()

            print(
                f"CARS {i + 1:02d}/{N} | {len(kept_idx):4d} variables retained | RMSEcv = {rmsecv:.5f}  ← Current best" if rmsecv == best_rmsecv else f"CARS {i + 1:02d}/{N} | {len(kept_idx):4d} variables retained | RMSEcv = {rmsecv:.5f}")

        print(f"\nCARS selection completed! Best subset contains {len(best_subset)} wavelength points, RMSEcv = {best_rmsecv:.5f}")
        return sorted(best_subset)

    def pls_analysis(self):
        if self.x_data is None or self.y_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return

        execution_time = 0.2
        start_time = time.time()

        try:
            self.start_pls_progress()

            dependent = int(self.dependent_var.get())
            polynomial = int(self.polynomial_var.get())
            train_ratio = float(self.train_ratio_var.get())

            X = self.y_data[:-dependent, :].T
            y = self.y_data[-dependent:, :].T

            selected_wavelengths = None
            if self.use_cars.get():
                print("Running CARS variable selection...")
                selected_idx = self.cars_variable_selection(
                    X, y,
                    N=100,
                    f=0.1,
                    cv_folds=5
                )
                selected_wavelengths = selected_idx
                X = X[:, selected_idx]
                print(f"Number of variables after CARS selection: {X.shape[1]}")

            if train_ratio <= 0 or train_ratio >= 1:
                messagebox.showwarning("Warning", "Training set ratio must be between 0 and 1")
                self.pls_running = False
                return

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                train_size=train_ratio,
                random_state=42,
                shuffle=True
            )

            train_data = np.column_stack([X_train, y_train])

            split_info = f"Data splitting completed:\nTraining set: {X_train.shape[0]} samples\nTest set: {X_test.shape[0]} samples\nTraining set ratio: {train_ratio * 100:.1f}%"
            self.listbox.insert(tk.END, split_info)
            self.listbox.insert(tk.END, "-" * 50)

            pls_analyzer = Linear(dependent, train_data, polynomial, X_test, y_test, selected_wavelengths)

            result_str = "PLS regression analysis completed\n"
            result_str += f"Number of dependent variables: {dependent}\n"
            result_str += f"Polynomial degree: {polynomial}\n"
            result_str += f"Execution time: {execution_time:.2f} 秒\n"
            result_str += "Detailed results saved to result.csv and verify.jpg"

            execution_time = time.time() - self.pls_start_time
            self.update_title(f"PLS analysis completed - Time elapsed {execution_time:.2f} s")

            messagebox.showinfo("Success", "PLS regression analysis completed\nResults saved to file.")

        except Exception as e:
            execution_time = time.time() - start_time
            self.update_title(f"PLS analysis failed - Time elapsed {execution_time:.1f}s")
            messagebox.showerror("Error", f"Error during PLS analysis: {str(e)}")

        finally:
            self.pls_running = False

class Linear:
    def __init__(self, dependent, train_data, polynomial=1, test_data=None, test_target=None, cars_selected=None):
        self.dependent = dependent
        self.train_data = train_data
        self.test_data = test_data
        self.test_target = test_target
        self.polynomial = polynomial
        self.cars_selected = cars_selected
        self.df = pd.DataFrame(train_data)
        self.n = len(self.df.columns) - self.dependent
        self.Polynomial()
        x0, y0, num, xishu, ch0, xish, sol = self.find()
        self.save(sol=sol)
        self.PLOT(ch0=ch0, num=num, x0=x0, y0=y0, xishu=xishu, xish=xish)

    def calculate_rmsecv(self, X, y, n_components):
        from sklearn.model_selection import cross_val_score

        k_folds = min(5, X.shape[0])
        cv_scores = []

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        rmse_list = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            pls = PLSRegression(n_components=n_components)
            pls.fit(X_train, y_train)

            y_pred = pls.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            rmse_list.append(rmse)

        return np.mean(rmse_list)

    def calculate_rmsep(self, X_train, y_train, X_test, y_test, n_components):
        if X_test is None or y_test is None:
            return None

        pls = PLSRegression(n_components=n_components)
        pls.fit(X_train, y_train)

        y_pred = pls.predict(X_test)

        rmsep = np.sqrt(mean_squared_error(y_test, y_pred))
        return rmsep

    def Polynomial(self):
        if self.polynomial != 1:
            temp = self.df.iloc[:, -self.dependent:]
            self.df.drop(self.df.columns[-self.dependent:], axis=1, inplace=True)
            count = self.n
            count_begin = 0
            for i in range(1, self.polynomial):
                count_end = count
                for k in range(self.n):
                    for j in range(count_begin, count_end):
                        name = "x" + str(k + 1) + str(j + 1) if i == 1 else "x" + str(k + 1) + self.df.columns[j][1:]
                        count += 1
                        self.df[name] = self.df.iloc[:, j].mul(self.df.iloc[:, k])
                count_begin = count_end
            for i in range(len(temp.columns)):
                self.df[temp.columns[i]] = temp.iloc[:, i]
            self.df.to_csv("changed.csv", encoding='GBK')

    def find(self):
        df = self.df
        df_matrix = np.array(df)
        mu = np.mean(df_matrix, axis=0)
        sig = np.std(df_matrix, axis=0)
        rr = df.corr()
        rr.to_csv("Correlation coefficient matrix.csv", encoding='GBK')
        data = preprocessing.scale(df_matrix)
        m = self.dependent
        n = len(df.columns) - m
        self.n = n
        x0 = df_matrix[:, :n]
        y0 = df_matrix[:, n:]
        e0 = data[:, :n]
        f0 = data[:, n:]
        num = len(df.iloc[:, 0])
        chg = np.identity(n)
        w = np.zeros([n, n])
        w_star = np.zeros([n, n])
        t = np.zeros([num, n])
        ss = []
        press = []
        Q_h2_list = []
        flag = 0

        max_components = min(10, num // 5)

        for i in range(n):

            if i >= max_components:
                print(f"[Overfitting Prevention] Maximum number of components limit {max_components} reached. Stopping forced.")
                flag = 1
                r = i - 1
                break

            matrix = e0.T @ f0 @ f0.T @ e0

            val, vec = np.linalg.eigh(matrix)
            idx = val.argsort()[::-1]
            w[:, i] = vec[:, idx[0]]

            w_star[:, i] = chg @ w[:, i]
            t[:, i] = e0 @ w[:, i]
            alpha = e0.T @ t[:, i] / (t[:, i].T @ t[:, i])
            p = alpha if alpha.ndim == 1 else alpha.flatten()
            chg = chg @ (np.identity(n) - np.outer(w[:, i], p))
            e0 = e0 - np.outer(t[:, i], p)
            beta_temp = np.linalg.pinv(np.c_[t[:, :i + 1], np.ones(num)]) @ f0
            beta_temp = np.delete(beta_temp, -1, axis=0)
            residual = f0 - t[:, :i + 1] @ beta_temp
            ss_current = np.sum(residual ** 2)
            ss.append(ss_current)

            if i == 0:
                press_i = ss_current * num / (num - 1)
                press.append(press_i)
                Q_h2_list.append(None)
                print(f"Component 1: SS = {ss_current:.6f}, PRESS ≈ {press_i:.6f}")
            else:
                press_i = ss_current * num / (num - i - 1)
                press.append(press_i)
                Q_h2 = 1 - press_i / ss[i - 1]
                Q_h2_list.append(Q_h2)

                print(f"Component {i + 1}: SS = {ss_current:.6f}, PRESS ≈ {press_i:.6f}, "
                      f"Q²_h² = {Q_h2:.4f}  (CumulativeQ² = {1 - press_i / ss[0]:.4f})")

                if Q_h2 < 0.0975:
                    print(f"Component {i + 1} Q²_h² = {Q_h2:.4f} < 0.0975, early stopping triggered!")
                    flag = 1
                    r = i
                    break

                if i >= 2 and Q_h2 < Q_h2_list[-2]:
                    print(f"Q²_h² started to decrease ({Q_h2_list[-2]:.4f} → {Q_h2:.4f}), early stopping to prevent overfitting")
                    flag = 1
                    #flag = 0
                    r = i
                    break
        if not flag:
            r = min(i, max_components - 1)
            print(f"Early stopping not triggered. Using the maximum allowed number of components: {r + 1}")
        else:
            print(f"Early stopping triggered. Final number of components used: {r + 1}")

        valid_q2 = [q for q in Q_h2_list if q is not None]
        best_q2 = max(valid_q2) if valid_q2 else 0.0
        final_q2 = Q_h2_list[r] if r < len(Q_h2_list) else Q_h2_list[-1] if Q_h2_list else 0.0

        print(f"Final number of components used: {r + 1}")
        print(f"Final Q²_h({r + 1}) = {final_q2:.4f}")
        print(f"Best historical Q² = {best_q2:.4f}")

        n_components_final = r + 1

        X_train = x0
        y_train = y0

        if X_train.shape[0] >= 5:
            self.rmsecv_values = []
            for i in range(self.dependent):
                rmsecv = self.calculate_rmsecv(X_train, y_train[:, i:i + 1], n_components_final)
                self.rmsecv_values.append(rmsecv)
                print(f"y{i + 1} RMSECV: {rmsecv:.4f}")
        else:
            self.rmsecv_values = [None] * self.dependent
            print("Insufficient number of samples to calculate RMSECV")

        if self.test_data is not None and self.test_target is not None:
            self.rmsep_values = []
            for i in range(self.dependent):
                rmsep = self.calculate_rmsep(X_train, y_train[:, i:i + 1],
                                             self.test_data, self.test_target[:, i:i + 1],
                                             n_components_final)
                self.rmsep_values.append(rmsep)
                print(f"y{i + 1} RMSEP: {rmsep:.4f}")
        else:
            self.rmsep_values = [None] * self.dependent
            print("No test set data available, unable to calculate RMSEP")

        t_final = t[:, :r + 1]
        beta_z = np.linalg.pinv(np.c_[t_final, np.ones(num)]) @ f0
        beta_z = np.delete(beta_z, -1, axis=0)
        xishu = w_star[:, :r + 1] @ beta_z

        mu_x = mu[:n]
        mu_y = mu[n:]
        sig_x = sig[:n]
        sig_y = sig[n:]

        ch0 = mu_y - np.sum((mu_x / sig_x)[:, None] * xishu, axis=0) * sig_y

        xish = xishu / sig_x[:, None] * sig_y[None, :]

        sol = np.r_[np.array([ch0]), xish]

        return x0, y0, num, xishu, ch0, xish, sol

    def PLOT(self, ch0, num, x0, y0, xishu, xish):
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 8,
            'axes.labelsize': 8,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 7,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'axes.linewidth': 0.5,
            'grid.linewidth': 0.5
        })

        plt.close('all')

        ch0 = np.tile(np.array(ch0), (num, 1))
        y_hat = ch0 + x0 @ xish

        i = 0

        fig_width_inch = 5.6 / 2.54
        fig_height_inch = 4.5 / 2.54
        fig, ax = plt.subplots(1, 1, figsize=(fig_width_inch, fig_height_inch))

        train_pred = y_hat[:, i].flatten()
        train_ref = y0[:, i].flatten()
        r_train = R2_func(train_pred, train_ref)
        rmsecv = self.rmsecv_values[i] if self.rmsecv_values[i] is not None else None

        has_test = False
        if self.test_data is not None and self.test_target is not None:
            test_pred = (ch0[0, i] + self.test_data @ xish[:, i:i + 1]).flatten()
            test_ref = self.test_target[:, i].flatten()
            r_test = R2_func(test_pred, test_ref)
            rmsep = self.rmsep_values[i] if self.rmsep_values[i] is not None else None
            has_test = True

        if has_test:
            all_x = np.concatenate([train_pred, test_pred])
            all_y = np.concatenate([train_ref, test_ref])
        else:
            all_x = train_pred
            all_y = train_ref

        global_min = min(all_x.min(), all_y.min())
        global_max = max(all_x.max(), all_y.max())
        margin = (global_max - global_min) * 0.05

        ax.plot([global_min - margin, global_max + margin],
                [global_min - margin, global_max + margin],
                'k-', linewidth=0.8)

        ax.scatter(train_pred, train_ref, marker='*', s=3, color='blue', alpha=0.7, label='Training')
        if has_test:
            ax.scatter(test_pred, test_ref, marker='s', s=3, color='red', alpha=0.7, label='Test')

        ax.set_xlabel("Predicted value(mg/Kg)")
        ax.set_ylabel("Reference value(mg/Kg)")
        ax.set_xlim(global_min - margin, global_max + margin)
        ax.set_ylim(global_min - margin, global_max + margin)
        ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)

        ax.text(0.02,0.98, '(k)', transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                fontsize=8, fontweight='bold')

        metrics = f"Training: $R$={r_train:.4f}"
        if rmsecv is not None:
            metrics += f"\nRMSECV={rmsecv:.4f}"
        if has_test:
            metrics += f"\nTest: $R$={r_test:.4f}"
            if rmsep is not None:
                metrics += f"\nRMSEP={rmsep:.4f}"

        ax.text(0.02, 0.85, metrics, transform=ax.transAxes,
                verticalalignment='top', fontsize=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

        if has_test:
            ax.legend(loc='lower right', fontsize=5)

        plt.savefig(
            "PLS_single.png",
            format='png',
            dpi=600,
            bbox_inches='tight',
            pad_inches=0.01,
            transparent=True,
            facecolor='none'
        )
        print("Saved: PLS_single.png")

        plt.show(block=False)

    def save(self, sol):
        sol = np.r_[[["y{}".format(i + 1) for i in range(self.dependent)]], sol]
        column_names = ["dependent", "x0"] + [f"x{i + 1}" for i in range(self.n)]
        sol = np.c_[column_names, sol]

        with open("result.csv", "w", newline="", encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerows(sol)

        if self.cars_selected is not None:
            with open("CARS_selected_wavelengths.txt", "w") as f:
                f.write("Selected wavelength indices (1-based):\n")
                f.write(",".join(map(str, [idx + 1 for idx in self.cars_selected])))


def R2_func(y_test, y):
    y_test = np.array(y_test).flatten()
    y = np.array(y).flatten()

    if len(y_test) > 1:
        r_value, p_value = pearsonr(y_test, y)
        return r_value
    else:
        return 0.0

if __name__ == "__main__":
    root = tk.Tk()
    app = Sample_preprocessing_GUI(root)
    root.mainloop()
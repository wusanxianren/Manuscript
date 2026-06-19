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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class Sample_preprocessing_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("异常样本剔除与PLS分析系统")
        self.root.geometry("1000x800")

        # 全局变量
        self.original_data = None  # 原始数据
        self.y_data = None  # 自变量
        self.x_data = None  # 因变量
        self.dependent_vars = 1  # 默认因变量个数
        self.polynomial_degree = 1  # 默认多项式次数
        self.pls_running = None

        self.do_smoothing = tk.BooleanVar(value=True)  # 默认打开三点平滑
        self.do_derivative = tk.BooleanVar(value=True)  # 默认打开一阶导数

        self.use_cars = tk.BooleanVar(value=False)  # 默认不使用 CARS

        # 创建GUI组件
        self.create_widgets()

    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=0, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        # 按钮
        self.load_btn = ttk.Button(button_frame, text="加载数据文件", command=self.load_data)
        self.load_btn.grid(row=0, column=0, padx=5)

        self.analyze_btn = ttk.Button(button_frame, text="光谱预处理", command=self.pca_analysis)
        self.analyze_btn.grid(row=0, column=1, padx=5)

        self.pls_btn = ttk.Button(button_frame, text="PLS回归", command=self.pls_analysis)
        self.pls_btn.grid(row=0, column=2, padx=5)

        self.export_btn = ttk.Button(button_frame, text="导出数据", command=self.export_data)
        self.export_btn.grid(row=0, column=3, padx=5)

        self.clear_btn = ttk.Button(button_frame, text="清除", command=self.clear_all)
        self.clear_btn.grid(row=0, column=4, padx=5)

        pls_frame = ttk.LabelFrame(main_frame, text="参数设置", padding="5")
        pls_frame.grid(row=1, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))

        ttk.Label(pls_frame, text="因变量个数:").grid(row=0, column=0, padx=5)
        self.dependent_var = tk.StringVar(value="1")
        self.dependent_entry = ttk.Entry(pls_frame, textvariable=self.dependent_var, width=10)
        self.dependent_entry.grid(row=0, column=1, padx=5)

        ttk.Label(pls_frame, text="异常检测阈值:").grid(row=0, column=2, padx=5)
        self.threshold_var = tk.StringVar(value="10.0")
        self.threshold_entry = ttk.Entry(pls_frame, textvariable=self.threshold_var, width=10)
        self.threshold_entry.grid(row=0, column=3, padx=5)

        ttk.Label(pls_frame, text="多项式次数:").grid(row=0, column=4, padx=5)
        self.polynomial_var = tk.StringVar(value="1")
        self.polynomial_entry = ttk.Entry(pls_frame, textvariable=self.polynomial_var, width=10)
        self.polynomial_entry.grid(row=0, column=5, padx=5)

        ttk.Label(pls_frame, text="训练集比例:").grid(row=0, column=6, padx=5)
        self.train_ratio_var = tk.StringVar(value="0.7")
        self.train_ratio_entry = ttk.Entry(pls_frame, textvariable=self.train_ratio_var, width=10)
        self.train_ratio_entry.grid(row=0, column=7, padx=5)

        ttk.Label(pls_frame, text="预处理:").grid(row=1, column=0, padx=(20, 5))

        ttk.Checkbutton(pls_frame, text="三点平滑", variable=self.do_smoothing).grid(row=1, column=1, padx=5)
        ttk.Checkbutton(pls_frame, text="一阶导数", variable=self.do_derivative).grid(row=1, column=2, padx=5)
        ttk.Checkbutton(pls_frame, text="使用 CARS 变量筛选", variable=self.use_cars).grid(row=1, column=3, padx=10)

        # 列表框和滚动条
        listbox_frame = ttk.Frame(main_frame)
        listbox_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 添加滚动条
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

        # 配置权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        listbox_frame.columnconfigure(0, weight=1)
        listbox_frame.rowconfigure(0, weight=1)

    def start_pls_progress(self):
        """开始PLS分析时调用，启动标题动态更新"""
        self.pls_start_time = time.time()
        self.pls_running = True
        self._update_progress_loop()

    def _update_progress_loop(self):
        # 内部递归函数，彻底取代原来的线程
        if not getattr(self, 'pls_running', False):   # 防止意外调用
            return

        elapsed = time.time() - self.pls_start_time
        self.update_title(f"PLS分析进行中... 已用时间: {elapsed:.1f}秒")

        # 500毫秒后再次调用自己
        self.root.after(500, self._update_progress_loop)

    def update_title(self, message):  # +++ 新增方法：更新窗口标题
        """更新窗口标题"""
        self.root.title(f"异常样本剔除与PLS分析系统 - {message}")

    def load_data(self):
        """加载数据文件"""
        file_path = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[("文本文件", "*.txt"),
            ("CSV文件", "*.csv"),
            ("Excel文件", "*.xlsx"),
            ("所有文件", "*.*")]
        )

        if file_path:
            try:
                # 根据文件扩展名选择加载方法
                if file_path.endswith('.txt'):
                    data = np.loadtxt(file_path)
                elif file_path.endswith('.csv'):
                    data = np.genfromtxt(file_path, delimiter=',')
                elif file_path.endswith('.xlsx'):
                    # 使用pandas读取Excel文件
                    df = pd.read_excel(file_path)
                    data = df.values  # 转换为numpy数组

                self.original_data = data
                self.y_data = data
                self.x_data = data

                # 显示在列表框中
                self.update_listbox(data)
                messagebox.showinfo("成功", f"数据加载成功！\n数据形状: {data.shape}")
                self.update_title("数据已加载")

            except Exception as e:
                messagebox.showerror("错误", f"加载数据时出错: {str(e)}")

    def export_data(self):
        """导出数据到文件"""
        if self.y_data is None:
            messagebox.showwarning("警告", "没有数据可导出！")
            return

        try:
            # 选择保存路径和文件格式
            file_path = filedialog.asksaveasfilename(
                title="导出数据",
                defaultextension=".txt",
                filetypes=[
                    ("文本文件", "*.txt"),
                    ("CSV文件", "*.csv"),
                    ("Excel文件", "*.xlsx"),
                    ("所有文件", "*.*")
                ]
            )

            if file_path:
                if file_path.endswith('.txt'):
                    # 导出为文本文件
                    np.savetxt(file_path, self.y_data, fmt='%.6f', delimiter='\t')
                elif file_path.endswith('.csv'):
                    # 导出为CSV文件
                    np.savetxt(file_path, self.y_data, fmt='%.6f', delimiter=',')
                elif file_path.endswith('.xlsx'):
                    # 导出为Excel文件
                    df = pd.DataFrame(self.y_data)
                    df.to_excel(file_path, index=False, header=False)

                messagebox.showinfo("成功", f"数据已成功导出到:\n{file_path}")
                self.update_title("数据已导出")

        except Exception as e:
            messagebox.showerror("错误", f"导出数据时出错: {str(e)}")

    def transpose_data(self):
        """转置数据"""
        if self.y_data is not None:
            self.y_data = self.y_data.T
            self.update_listbox(self.y_data)
        else:
            messagebox.showwarning("警告", "请先加载数据！")

    def pca_analysis(self):
        """主成分分析"""
        if self.original_data is None:
            messagebox.showwarning("警告", "请先加载数据！")
            return

        try:
            self.update_title("正在进行主成分分析...")
            self.root.update()

            self.dependent_vars = int(self.dependent_var.get())  # 获取因变量参数

            data_T = self.original_data.T
            self.y_data = data_T[:, :-self.dependent_vars].astype(float)  # 光谱
            self.x_data = data_T[:, -self.dependent_vars:].astype(float)  # 含量

            threshold = float(self.threshold_var.get())

            x = self.y_data.astype(float) # 获取数据

            n = x.shape[0]  # 样本数量

            # 数据标准化
            scaler = StandardScaler()
            sr = scaler.fit_transform(self.y_data)

            # 主成分分析
            pca = PCA()
            pca.fit(sr)

            pcs = pca.components_.T  # 主成分
            newdata = pca.transform(sr)

            variances = pca.explained_variance_  # 方差

            g = variances / np.sum(variances) # 计算贡献率

            d = np.zeros(n) # 计算加权主成分得分

            n_components = min(3, pcs.shape[1]) # 确保至少有3个主成分可用

            for i in range(n):
                score = 0
                for j in range(n_components):
                    projection = np.dot(x[i, :], pcs[:, j])
                    score += g[j] * projection
                d[i] = score

            # 标记异常点
            upper_threshold = np.mean(d) + threshold
            lower_threshold = np.mean(d) - threshold
            outliers = np.where((d > upper_threshold) | (d < lower_threshold))[0]

            self.create_outlier_plot(d, threshold)

            # 去除异常样本并更新数据
            if len(outliers) > 0:
                # 去除异常样本
                self.y_data = x
                self.y_data = np.column_stack([self.y_data, self.x_data])
                cleaned_data = np.delete(self.y_data, outliers, axis=0)

                # 更新数据
                self.y_data = cleaned_data

                # 显示结果
                outlier_text = f"检测到 {len(outliers)} 个异常样本:\n" + ", ".join([f"样本{i + 1}" for i in outliers])
                result_text = f"{outlier_text}\n\n已自动从数据中移除这些异常样本。\n数据从 {n} 个样本减少到 {len(cleaned_data)} 个样本。"
                messagebox.showinfo("异常样本检测与清理", result_text)
            else:
                self.y_data = x
                self.y_data = np.column_stack([self.y_data, self.x_data])

                messagebox.showinfo("异常样本检测", "未检测到异常样本，数据保持原样。")

            self.transpose_data()  # 转置数据
            self.x_data = self.y_data[-self.dependent_vars:, :]  # 含量
            self.y_data = self.y_data[:-self.dependent_vars, :]  # 光谱
            self.smooth_derivative_combined() # 平滑和导数处理

            self.y_data = np.vstack([self.y_data, self.x_data])
            self.update_listbox(self.y_data)

            self.update_title("主成分分析完成")

        except Exception as e:
            messagebox.showerror("错误", f"主成分分析时出错: {str(e)}")

    def create_outlier_plot(self, d, threshold):
        """创建异常检测图"""
        fig, ax = plt.subplots(figsize=(12, 6))

        n = len(d)
        indices = np.arange(1, n + 1)
        mean_d = np.mean(d)

        # 绘制主成分得分
        ax.plot(indices, d, '*r', markersize=8, label='主成分得分', alpha=0.7)

        # 绘制阈值线
        x_range = np.linspace(0.5, n + 0.5, 100)
        ax.plot(x_range, np.full(100, mean_d + threshold), 'r-',
                linewidth=2, label='阈值上限')
        ax.plot(x_range, np.full(100, mean_d - threshold), 'r-',
                linewidth=2, label='阈值下限')
        ax.axhline(y=mean_d, color='b', linestyle='--', alpha=0.7, label='平均值')

        # 标记异常点
        outliers = np.where((d > mean_d + threshold) | (d < mean_d - threshold))[0]
        if len(outliers) > 0:
            ax.plot(indices[outliers], d[outliers], 'ko', markersize=10,
                    fillstyle='none', markeredgewidth=2, label='异常点')

            # 添加异常点标签
            for i in outliers:
                ax.annotate(f'{i + 1}', (indices[i], d[i]),
                            xytext=(5, 5), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        ax.set_xlabel('样本索引', fontsize=12)
        ax.set_ylabel('主成分得分', fontsize=12)
        ax.set_title('异常样本剔除——主成分得分分析', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, n + 0.5)

        # 显示图形
        plt.tight_layout()
        plt.show(block=False)

    def clear_all(self):
        """清除所有数据"""
        self.y_data = None
        self.listbox.delete(0, tk.END)
        self.update_title("数据已清除")

    def update_listbox(self, data):
        """更新列表框显示"""
        self.listbox.delete(0, tk.END)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # 格式化显示数据
        for i in range(data.shape[0]):
            row_str = "  ".join([f"{val:.6f}" for val in data[i, :]])
            self.listbox.insert(tk.END, row_str)

    def first_derivative(self, data):
        """一阶导数计算实现"""
        return np.diff(data, axis=0)

    def three_point_smoothing(self, data):
        """三点平滑处理实现"""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        n_rows, n_cols = data.shape
        smoothed = np.zeros_like(data)

        for col in range(n_cols):
            # 保持首尾点不变
            smoothed[0, col] = data[0, col]
            smoothed[-1, col] = data[-1, col]

            # 对中间点进行三点平滑
            for i in range(1, n_rows - 1):
                smoothed[i, col] = (data[i - 1, col] + data[i, col] + data[i + 1, col]) / 3.0

        return smoothed

    def smooth_derivative_combined(self):
        """平滑和导数组合处理"""
        if self.y_data is None:
            messagebox.showwarning("警告", "请先加载数据！")
            return

        try:
            data = self.y_data.copy()  # 防止后续操作影响原始光谱数据

            if self.do_smoothing.get():
                data = self.three_point_smoothing(data)

            if self.do_derivative.get():
                if data.shape[0] < 2:
                    messagebox.showwarning("警告", "数据行数太少，无法计算一阶导数")
                    return
                data = self.first_derivative(data)

            self.y_data = data
            self.update_listbox(data)
        except Exception as e:
            messagebox.showerror("错误", f"平滑求导处理时出错: {str(e)}")

    def cars_variable_selection(self, X, y, N=50, f=0.1, cv_folds=5, max_components=15):
        """
        CARS 变量筛选（经典实现）
        参数：
            X           : (n_samples, n_features) 光谱矩阵
            y           : (n_samples, n_targets)   目标浓度（支持多因变量，取第一列即可）
            N           : 蒙特卡洛采样次数（默认50）
            f           : 每次保留的变量比例（指数衰减）
            cv_folds    : 交叉验证折数
            max_components : PLS 最大主成分数
        返回：
            selected_idx : 被选中的变量索引列表（从0开始）
        """
        if y.ndim == 2:
            y = y[:, 0] if y.shape[1] > 1 else y.ravel()

        n_samples, n_vars = X.shape
        best_rmsecv = np.inf
        best_subset = None
        rmsecv_history = []

        print(f"开始 CARS 变量筛选（N={N}，样本数={n_samples}，变量数={n_vars}）...")

        prev_kept = None  # 用于记录上一轮选中的变量

        for i in range(N):
            # 1. 正确的指数衰减：从全部变量 → 2个（经典CARS公式）
            ratio = np.exp(np.log(2.0 / n_vars) * i / (N - 1))
            n_keep = max(2, int(n_vars * ratio + 0.5))  # +0.5 更接近round

            # 2. 自适应加权采样（关键修复：所有变量都要有概率！）
            if i == 0:
                kept_idx = np.random.choice(n_vars, size=n_keep, replace=False)
            else:
                # 在上一轮变量上建模
                n_comp_temp = min(max_components, len(prev_kept), n_samples - 2)
                n_comp_temp = max(1, n_comp_temp)
                pls_temp = PLSRegression(n_components=n_comp_temp)
                pls_temp.fit(X[:, prev_kept], y)
                coef = np.abs(pls_temp.coef_.ravel())

                # 权重分配给全部变量（未入选为0）
                weights = np.zeros(n_vars)
                weights[prev_kept] = coef
                weights += 1e-8  # 防止概率全为0
                weights /= weights.sum()

                kept_idx = np.random.choice(n_vars, size=n_keep, replace=False, p=weights)

            prev_kept = kept_idx.copy()

            # 3. 稳健的交叉验证（防止崩溃）
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
                f"CARS {i + 1:02d}/{N} | 保留 {len(kept_idx):4d} 个变量 | RMSEcv = {rmsecv:.5f}  ← 当前最佳" if rmsecv == best_rmsecv else f"CARS {i + 1:02d}/{N} | 保留 {len(kept_idx):4d} 个变量 | RMSEcv = {rmsecv:.5f}")

        print(f"\nCARS 筛选完成！最佳子集包含 {len(best_subset)} 个波长点，RMSEcv = {best_rmsecv:.5f}")
        return sorted(best_subset)  # 返回排序后的索引（1-based后面再加1）

    def pls_analysis(self):
        """PLS回归分析"""
        if self.x_data is None or self.y_data is None:
            messagebox.showwarning("警告", "请先加载数据！")
            return

        execution_time = 0.2
        start_time = time.time()

        try:
            self.start_pls_progress()

            # 获取参数
            dependent = int(self.dependent_var.get())
            polynomial = int(self.polynomial_var.get())
            train_ratio = float(self.train_ratio_var.get())

            # 数据准备
            X = self.y_data[:-dependent, :].T  # 光谱
            y = self.y_data[-dependent:, :].T  # 含量

            # CARS 变量筛选
            selected_wavelengths = None
            if self.use_cars.get():
                print("正在执行 CARS 变量筛选...")
                selected_idx = self.cars_variable_selection(
                    X, y,
                    N=100,  # 可自行调节
                    f=0.1,
                    cv_folds=5
                )
                selected_wavelengths = selected_idx
                X = X[:, selected_idx]  # 只保留选中的波长
                print(f"CARS 筛选后变量数：{X.shape[1]}")

            # 检查训练集比例是否合理
            if train_ratio <= 0 or train_ratio >= 1:
                messagebox.showwarning("警告", "训练集比例必须在0和1之间！")
                self.pls_running = False
                return

            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                train_size=train_ratio,
                random_state=42,
                shuffle=True
            )

            train_data = np.column_stack([X_train, y_train])

            # 显示分割信息
            split_info = f"数据分割完成:\n训练集: {X_train.shape[0]} 样本\n测试集: {X_test.shape[0]} 样本\n训练集比例: {train_ratio * 100:.1f}%"
            self.listbox.insert(tk.END, split_info)
            self.listbox.insert(tk.END, "-" * 50)

            # 使用Linear类进行PLS分析
            pls_analyzer = Linear(dependent, train_data, polynomial, X_test, y_test, selected_wavelengths)

            # 获取分析结果
            result_str = "PLS回归分析完成！\n"
            result_str += f"因变量个数: {dependent}\n"
            result_str += f"多项式次数: {polynomial}\n"
            result_str += f"执行时间: {execution_time:.2f} 秒\n"
            result_str += "详细结果已保存到 result.csv 和 verify.jpg"

            execution_time = time.time() - self.pls_start_time
            self.update_title(f"PLS分析完成 - 用时 {execution_time:.2f} 秒")

            messagebox.showinfo("成功", "PLS回归分析完成！\n结果已保存到文件。")

        except Exception as e:
            execution_time = time.time() - start_time
            self.update_title(f"PLS分析失败 - 用时 {execution_time:.1f}秒")
            messagebox.showerror("错误", f"PLS分析时出错: {str(e)}")

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
        """计算RMSECV（交叉验证均方根误差）"""
        from sklearn.model_selection import cross_val_score

        # 使用k-fold交叉验证
        k_folds = min(5, X.shape[0])  # 根据样本数选择折数
        cv_scores = []

        # 使用KFold手动计算
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        rmse_list = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 创建PLS模型
            pls = PLSRegression(n_components=n_components)
            pls.fit(X_train, y_train)

            # 预测
            y_pred = pls.predict(X_val)

            # 计算RMSE
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            rmse_list.append(rmse)

        return np.mean(rmse_list)

    def calculate_rmsep(self, X_train, y_train, X_test, y_test, n_components):
        """计算RMSEP（预测均方根误差）"""
        if X_test is None or y_test is None:
            return None

        # 在训练集上训练模型
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_train, y_train)

        # 在测试集上预测
        y_pred = pls.predict(X_test)

        # 计算RMSE
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
        rr.to_csv("相关系数矩阵.csv", encoding='GBK')
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
                print(f"【防过拟合】已达到最大成分数限制 {max_components}，强制停止")
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
                Q_h2_list.append(None)  # 第一个成分前无Q²
                print(f"成分 1: SS = {ss_current:.6f}, PRESS ≈ {press_i:.6f}")
            else:
                press_i = ss_current * num / (num - i - 1)
                press.append(press_i)
                Q_h2 = 1 - press_i / ss[i - 1]
                Q_h2_list.append(Q_h2)

                print(f"成分 {i + 1}: SS = {ss_current:.6f}, PRESS ≈ {press_i:.6f}, "
                      f"Q²_h² = {Q_h2:.4f}  (累计Q² = {1 - press_i / ss[0]:.4f})")

                if Q_h2 < 0.0975:
                    print(f"第 {i + 1} 个成分 Q²_h² = {Q_h2:.4f} < 0.0975，触发早停！")
                    flag = 1
                    r = i
                    break

                # 附加保护：Q²开始下降也停止（非常有效防过拟合）
                if i >= 2 and Q_h2 < Q_h2_list[-2]:
                    print(f"Q²_h² 开始下降 ({Q_h2_list[-2]:.4f} → {Q_h2:.4f})，提前停止防过拟合")
                    flag = 1
                    #flag = 0
                    r = i
                    break
        if not flag:
            r = min(i, max_components - 1)
            print(f"未触发早停，使用最大允许成分数: {r + 1}")
        else:
            print(f"早停触发，最终使用成分数: {r + 1}")

        valid_q2 = [q for q in Q_h2_list if q is not None]
        best_q2 = max(valid_q2) if valid_q2 else 0.0
        final_q2 = Q_h2_list[r] if r < len(Q_h2_list) else Q_h2_list[-1] if Q_h2_list else 0.0

        print(f"最终使用成分数: {r + 1}")
        print(f"最终Q²_h({r + 1}) = {final_q2:.4f}")
        print(f"历史最佳Q² = {best_q2:.4f}")

        # 计算RMSECV和RMSEP
        n_components_final = r + 1

        # 提取训练数据
        X_train = x0
        y_train = y0

        # 计算RMSECV
        if X_train.shape[0] >= 5:  # 需要至少5个样本进行交叉验证
            self.rmsecv_values = []
            for i in range(self.dependent):
                rmsecv = self.calculate_rmsecv(X_train, y_train[:, i:i + 1], n_components_final)
                self.rmsecv_values.append(rmsecv)
                print(f"y{i + 1} RMSECV: {rmsecv:.4f}")
        else:
            self.rmsecv_values = [None] * self.dependent
            print("样本数不足，无法计算RMSECV")

        # 计算RMSEP
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
            print("无测试集数据，无法计算RMSEP")

        # 提取最终模型
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
        # === 设置 Food Control 合规样式 ===
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

        plt.close('all')  # 关闭旧图

        ch0 = np.tile(np.array(ch0), (num, 1))
        y_hat = ch0 + x0 @ xish  # 训练集预测值

        # 只处理第一个因变量 (i=0)
        i = 0

        fig_width_inch = 5.6 / 2.54
        fig_height_inch = 4.5 / 2.54
        fig, ax = plt.subplots(1, 1, figsize=(fig_width_inch, fig_height_inch))

        # --- 训练集 ---
        train_pred = y_hat[:, i].flatten()
        train_ref = y0[:, i].flatten()
        r_train = R2_func(train_pred, train_ref)
        rmsecv = self.rmsecv_values[i] if self.rmsecv_values[i] is not None else None

        # --- 测试集 ---
        has_test = False
        if self.test_data is not None and self.test_target is not None:
            test_pred = (ch0[0, i] + self.test_data @ xish[:, i:i + 1]).flatten()
            test_ref = self.test_target[:, i].flatten()
            r_test = R2_func(test_pred, test_ref)
            rmsep = self.rmsep_values[i] if self.rmsep_values[i] is not None else None
            has_test = True

        # --- 坐标范围 ---
        if has_test:
            all_x = np.concatenate([train_pred, test_pred])
            all_y = np.concatenate([train_ref, test_ref])
        else:
            all_x = train_pred
            all_y = train_ref

        global_min = min(all_x.min(), all_y.min())
        global_max = max(all_x.max(), all_y.max())
        margin = (global_max - global_min) * 0.05

        # 1:1 线
        ax.plot([global_min - margin, global_max + margin],
                [global_min - margin, global_max + margin],
                'k-', linewidth=0.8)

        # 散点
        ax.scatter(train_pred, train_ref, marker='*', s=3, color='blue', alpha=0.7, label='Training')
        if has_test:
            ax.scatter(test_pred, test_ref, marker='s', s=3, color='red', alpha=0.7, label='Test')

        # 坐标轴
        ax.set_xlabel("Predicted value(mg/Kg)")
        ax.set_ylabel("Reference value(mg/Kg)")
        ax.set_xlim(global_min - margin, global_max + margin)
        ax.set_ylim(global_min - margin, global_max + margin)
        ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)

        ax.text(0.02,0.98, '(k)', transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                fontsize=8, fontweight='bold')

        # 指标文本
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

        # 保存为单张 PDF
        plt.savefig(
            "PLS_single.png",
            format='png',
            dpi=600,
            bbox_inches='tight',
            pad_inches=0.01,
            transparent=True,  # 透明背景
            facecolor='none'  # 确保无背景色
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
    # 计算相关系数R
    y_test = np.array(y_test).flatten()
    y = np.array(y).flatten()

    # 计算相关系数
    if len(y_test) > 1:
        r_value, p_value = pearsonr(y_test, y)
        return r_value
    else:
        return 0.0

if __name__ == "__main__":
    root = tk.Tk()
    app = Sample_preprocessing_GUI(root)
    root.mainloop()
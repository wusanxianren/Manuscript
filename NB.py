import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# --- 添加中文字体配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# --- 1. 加载你自己的数据 ---
df = pd.read_excel('核心-光谱.xlsx', header=0)

# 假设第一列是目标变量，其余是特征
X = df.iloc[:, 1:].values  # 特征
y_series = df.iloc[:, 0]  # 目标变量

# 获取原始特征名称
feature_names = df.columns[1:].tolist()

# 尝试将 y 转换为数值，方便绘图
try:
    y = y_series.astype('category').cat.codes.values
    target_names = y_series.astype('category').cat.categories.tolist()
except Exception:
    print("警告: 目标变量似乎不是分类标签，将使用连续值或颜色映射。")
    y = y_series.values
    target_names = ["Unknown"]

print(f"原始数据形状: {X.shape}")
print(f"特征名称: {feature_names}")
print(f"目标类别: {target_names}")
print(f"各类别样本数量: {pd.Series(y).value_counts().to_dict()}")

# --- 2. 标准化 ---
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. 执行 PCA 并保留 3 个主成分 ---
n_components_3d = 3
pca_3d = PCA(n_components=n_components_3d)
X_pca_3d = pca_3d.fit_transform(X_scaled)

# --- 4. 查看 PCA 结果 ---
print(f"\n--- 3D PCA 结果 ---")
print(f"降维后数据形状: {X_pca_3d.shape}")
print(f"各主成分的方差解释比例: {pca_3d.explained_variance_ratio_}")
print(f"累积方差解释比例: {np.sum(pca_3d.explained_variance_ratio_):.4f}")
print(f"主成分载荷矩阵形状: {pca_3d.components_.shape}")
print(f"主成分载荷矩阵 (PC1, PC2, PC3): \n{pca_3d.components_}")

# --- 计算每个原始特征的最终贡献度 ---
# components_ 的 shape 是 (n_components, n_features)
# 我们需要对每一列（每个原始特征）计算其在所有选定主成分中载荷的绝对值之和
absolute_loadings = np.abs(pca_3d.components_) # 计算所有选定主成分载荷的绝对值
feature_contributions = np.sum(absolute_loadings, axis=0) # 沿着主成分轴 (axis=0) 求和

# 创建结果 DataFrame
contributions_df = pd.DataFrame({
    'Feature': feature_names,
    'Contribution_Score': feature_contributions
}).sort_values(by='Contribution_Score', ascending=False) # 按贡献度降序排列

print(f"\n--- 每个原始特征的最终贡献度 (前 {n_components_3d} 个主成分) ---")
print(contributions_df)

# --- 5. 3D 可视化 ---
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'pdf.fonttype': 42,  # 嵌入字体（关键！）
    'ps.fonttype': 42,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.5
})

# 创建图形（双栏宽度 17.6 cm）
fig_width_inch = 17.6 / 2.54  # = 6.93 inch
fig_height_inch = 12.0 / 2.54  # 高度 ~12 cm（适合 3D 图）
fig = plt.figure(figsize=(fig_width_inch, fig_height_inch))
ax = fig.add_subplot(111, projection='3d')

# === 设置白色背景（期刊要求）===
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('k')  # 坐标轴 pane 边框设为黑色
ax.yaxis.pane.set_edgecolor('k')
ax.zaxis.pane.set_edgecolor('k')
ax.xaxis.pane.set_alpha(0.1)  # 微透明，避免遮挡
ax.yaxis.pane.set_alpha(0.1)
ax.zaxis.pane.set_alpha(0.1)

# 固定视角
ax.view_init(elev=45, azim=45)  # 调整仰角更清晰

# 绘图逻辑（保持不变，但确保 target_names 为英文）
if y is not None and len(np.unique(y)) > 1:
    unique_y = np.unique(y)
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_y)))

    for i, class_val in enumerate(unique_y):
        mask = y == class_val
        x_data = X_pca_3d[mask, 0]
        y_data = X_pca_3d[mask, 1]
        z_data = X_pca_3d[mask, 2]
        color = colors[i]
        marker = markers[i % len(markers)]

        ax.scatter(x_data, y_data, z_data,
                   c=[color], marker=marker, edgecolors='k', s=20, alpha=0.9,
                   label=target_names[class_val] if target_names != ["Unknown"] else f'Class {class_val}')

    ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})')
    ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})')
    ax.legend(title="Class", loc='upper left', frameon=True, fancybox=False, shadow=False)  # 英文 "Class"

else:
    ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
               c='steelblue', edgecolors='k', s=20, alpha=0.9)
    ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})')
    ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})')
    ax.set_title('3D PCA Visualization (No Class Info)', fontsize=9, fontweight='bold')

ax.tick_params(axis='x', pad=-2)  # 调整 x 轴标签距离
ax.tick_params(axis='y', pad=-2)  # 调整 y 轴标签距离
ax.tick_params(axis='z', pad=-2)  # 调整 z 轴标签距离

# === 添加左上角 (a) 标注 ===
# 获取坐标轴范围
xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
ax.text(xlim[1], ylim[0] + 0.05, zlim[1] + 0.05, '(b)',  # x最小, y最大, z最大 = 左上角
        verticalalignment='top', horizontalalignment='left',
        fontsize=8, fontweight='bold')

# === 保存为 PNG===
plt.tight_layout(pad=2.0)  # 为 3D 图留更多边距

output_png = "3D_PCA_Visualization.png"
plt.savefig(
    output_png,
    format='png',
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.5,
    transparent=False,
    facecolor='white'
)
print(f"✅ 透明背景 PNG 已生成: {output_png}")

plt.show()

# --- 6. 构建朴素贝叶斯产区识别模型 ---  # <--- 【全新模块：第6部分】
print("\n" + "=" * 60)
print("开始构建朴素贝叶斯产区识别模型")
print("=" * 60)

# 准备完整的数据集用于建模

pca_for_model = PCA(n_components=0.95)  # 保留 95% 的方差
X_pca_model = pca_for_model.fit_transform(X_scaled)

print(f"PCA 降维后特征数: {X_pca_model.shape[1]}")
print(f"累计解释方差比例: {pca_for_model.explained_variance_ratio_.sum():.4f}")

y_full = y  # 目标变量

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(
    X_pca_model, y_full, test_size=0.3, stratify=y_full
)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")
print(f"训练集中各类别分布: {pd.Series(y_train).value_counts().to_dict()}")
print(f"测试集中各类别分布: {pd.Series(y_test).value_counts().to_dict()}")

# 创建朴素贝叶斯模型
nb_model = GaussianNB()

# 训练模型
nb_model.fit(X_train, y_train)

# 预测
y_pred = nb_model.predict(X_test)
y_pred_proba = nb_model.predict_proba(X_test)

# 计算准确率
train_accuracy = nb_model.score(X_train, y_train)
test_accuracy = nb_model.score(X_test, y_test)

print(f"\n--- 朴素贝叶斯模型性能 ---")
print(f"训练集准确率: {train_accuracy:.4f}")
print(f"测试集准确率: {test_accuracy:.4f}")

# 详细的分类报告
print(f"\n--- 分类报告 ---")
report = classification_report(y_test, y_pred,
                               target_names=[str(name) for name in target_names])
print(report)

# --- 7. 交叉验证评估 ---
cv_scores = cross_val_score(nb_model, X_pca_model, y_full, cv=5, scoring='accuracy')
print(f"\n--- 5折交叉验证结果 ---")
print(f"各折准确率: {cv_scores}")
print(f"平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# --- 8. 混淆矩阵可视化 ---
plt.figure(figsize=(fig_width_inch, fig_height_inch))

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names,
            cbar_kws={'shrink': 0.8})  # 调整色条大小

# 坐标轴（英文）
plt.xlabel('Predicted labels')
plt.ylabel('Actual labels')

plt.text(-0.05, 0.98, '(b)', transform=plt.gca().transAxes,
         verticalalignment='top', horizontalalignment='left',
         fontsize=8, fontweight='bold')

# 紧凑布局
plt.tight_layout(pad=1.0)

# === 保存为 PNG ===
plt.savefig(
    'Confusion_Matrix.png',
    format='png',
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.02,
    transparent=True,
    facecolor='none'
)
plt.show()

# --- 9. 特征重要性分析（基于PCA载荷）---
print(f"\n--- 基于PCA载荷的特征重要性分析 ---")
# 使用所有主成分来计算特征重要性
pca_all = PCA()  # 不限制组件数量
X_pca_all = pca_all.fit_transform(X_scaled)

# 计算每个原始特征对分类的总体贡献
# 这里我们考虑所有主成分，权重为对应的方差解释比例
all_loadings = np.abs(pca_all.components_)  # 形状: (min(n_samples, n_features), n_features)
variance_weights = pca_all.explained_variance_ratio_

# 计算加权特征重要性
weighted_importance = np.zeros(len(feature_names))
for i in range(min(len(variance_weights), len(all_loadings))):
    weighted_importance += all_loadings[i] * variance_weights[i]

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': weighted_importance
}).sort_values(by='Importance', ascending=False)

print(f"基于PCA载荷的特征重要性排序:")
print(feature_importance_df)

# --- 10. 预测概率分析 ---
print(f"\n--- 预测概率分析 ---")
prob_df = pd.DataFrame(y_pred_proba, columns=[f"P({name})" for name in target_names])
prob_df['True_Label'] = [target_names[i] for i in y_test]
prob_df['Predicted_Label'] = [target_names[i] for i in y_pred]

print("测试集预测概率示例（前10个样本）:")
print(prob_df.head(10))


# --- 11. 模型保存和预测函数 ---
def predict_origin(features):
    """
    预测产地的函数

    Parameters:
    features: array-like, shape = [n_samples, n_features]
        输入特征数据

    Returns:
    predictions: array of predicted classes
    probabilities: array of prediction probabilities
    """
    # 标准化输入特征
    features_scaled = scaler.transform(features)
    # 预测
    pred_classes = nb_model.predict(features_scaled)
    pred_proba = nb_model.predict_proba(features_scaled)

    return pred_classes, pred_proba


# --- 13. 预测结果统计 ---
print(f"\n--- 预测结果详细统计 ---")
comparison_df = pd.DataFrame({
    'Sample_Index': range(len(y_test)),
    'True_Label': [target_names[i] for i in y_test],
    'Predicted_Label': [target_names[i] for i in y_pred],
    'Is_Correct': y_test == y_pred,
    'Confidence': np.max(y_pred_proba, axis=1)
})

print("预测结果统计:")
print(comparison_df)

print(f"\n各产地预测准确率:")
for i, class_name in enumerate(target_names):
    class_mask = y_test == i
    if np.any(class_mask):
        class_correct = np.sum((y_test == i) & (y_pred == i))
        class_total = np.sum(y_test == i)
        class_acc = class_correct / class_total
        print(f"{class_name}: {class_correct}/{class_total} = {class_acc:.3f}")

print(f"\n--- 朴素贝叶斯产区识别模型构建完成 ---")
print(f"模型已准备好用于新的产地识别任务！")
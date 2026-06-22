import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel('Core-Spectrum.xlsx', header=0)

X = df.iloc[:, 1:].values
y_series = df.iloc[:, 0]

feature_names = df.columns[1:].tolist()

try:
    y = y_series.astype('category').cat.codes.values
    target_names = y_series.astype('category').cat.categories.tolist()
except Exception:
    y = y_series.values
    target_names = ["Unknown"]

print(f"Shape of Raw Data: {X.shape}")
print(f"Feature Names: {feature_names}")
print(f"Target Class: {target_names}")
print(f"Number of Samples per Class: {pd.Series(y).value_counts().to_dict()}")

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

n_components_3d = 3
pca_3d = PCA(n_components=n_components_3d)
X_pca_3d = pca_3d.fit_transform(X_scaled)

print(f"\n--- 3D PCA Results ---")
print(f"Shape of Reduced Data: {X_pca_3d.shape}")
print(f"Variance Explained by Each Principal Component: {pca_3d.explained_variance_ratio_}")
print(f"Cumulative Variance Explained: {np.sum(pca_3d.explained_variance_ratio_):.4f}")
print(f"Shape of the Loading Matrix: {pca_3d.components_.shape}")
print(f"PCA Loading Matrix (PC1, PC2, PC3): \n{pca_3d.components_}")

absolute_loadings = np.abs(pca_3d.components_)
feature_contributions = np.sum(absolute_loadings, axis=0)

contributions_df = pd.DataFrame({
    'Feature': feature_names,
    'Contribution_Score': feature_contributions
}).sort_values(by='Contribution_Score', ascending=False)

print(f"\n--- Final Contribution of Each Original Feature (First {n_components_3d} Principal Components) ---")
print(contributions_df)

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.5
})

fig_width_inch = 17.6 / 2.54
fig_height_inch = 12.0 / 2.54
fig = plt.figure(figsize=(fig_width_inch, fig_height_inch))
ax = fig.add_subplot(111, projection='3d')

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('k')
ax.yaxis.pane.set_edgecolor('k')
ax.zaxis.pane.set_edgecolor('k')
ax.xaxis.pane.set_alpha(0.1)
ax.yaxis.pane.set_alpha(0.1)
ax.zaxis.pane.set_alpha(0.1)

ax.view_init(elev=45, azim=45)

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
    ax.legend(title="Class", loc='upper left', frameon=True, fancybox=False, shadow=False)

else:
    ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
               c='steelblue', edgecolors='k', s=20, alpha=0.9)
    ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})')
    ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})')
    ax.set_title('3D PCA Visualization (No Class Info)', fontsize=9, fontweight='bold')

ax.tick_params(axis='x', pad=-2)
ax.tick_params(axis='y', pad=-2)
ax.tick_params(axis='z', pad=-2)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
ax.text(xlim[1], ylim[0] + 0.05, zlim[1] + 0.05, '(b)',
        verticalalignment='top', horizontalalignment='left',
        fontsize=8, fontweight='bold')

plt.tight_layout(pad=2.0)

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
print(f" PNG: {output_png}")

plt.show()

pca_for_model = PCA(n_components=0.95)
X_pca_model = pca_for_model.fit_transform(X_scaled)

print(f"Number of Features After PCA Reduction: {X_pca_model.shape[1]}")
print(f"Cumulative Proportion of Variance Explained: {pca_for_model.explained_variance_ratio_.sum():.4f}")

y_full = y

X_train, X_test, y_train, y_test = train_test_split(
    X_pca_model, y_full, test_size=0.3, stratify=y_full
)

print(f"Training Set Size: {X_train.shape}")
print(f"Test Set Size: {X_test.shape}")
print(f"Class Distribution in the Training Set: {pd.Series(y_train).value_counts().to_dict()}")
print(f"Class Distribution in the Test Set: {pd.Series(y_test).value_counts().to_dict()}")

nb_model = GaussianNB()

nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)
y_pred_proba = nb_model.predict_proba(X_test)

train_accuracy = nb_model.score(X_train, y_train)
test_accuracy = nb_model.score(X_test, y_test)

print(f"\n--- Naive Bayes Model Performance ---")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print(f"\n--- Classification Report ---")
report = classification_report(y_test, y_pred,
                               target_names=[str(name) for name in target_names])
print(report)

cv_scores = cross_val_score(nb_model, X_pca_model, y_full, cv=5, scoring='accuracy')
print(f"\n--- 5-Fold Cross-Validation Results ---")
print(f"Accuracy per Fold: {cv_scores}")
print(f"Average Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

plt.figure(figsize=(fig_width_inch, fig_height_inch))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names,
            cbar_kws={'shrink': 0.8})

plt.xlabel('Predicted labels')
plt.ylabel('Actual labels')

plt.text(-0.05, 0.98, '(b)', transform=plt.gca().transAxes,
         verticalalignment='top', horizontalalignment='left',
         fontsize=8, fontweight='bold')

plt.tight_layout(pad=1.0)

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

print(f"\n--- Feature Importance Analysis Based on PCA Loadings ---")
pca_all = PCA()
X_pca_all = pca_all.fit_transform(X_scaled)

all_loadings = np.abs(pca_all.components_)
variance_weights = pca_all.explained_variance_ratio_

weighted_importance = np.zeros(len(feature_names))
for i in range(min(len(variance_weights), len(all_loadings))):
    weighted_importance += all_loadings[i] * variance_weights[i]

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': weighted_importance
}).sort_values(by='Importance', ascending=False)

print(f"The feature importance ranking based on PCA loadings:")
print(feature_importance_df)

print(f"\n--- Predictive probability analysis ---")
prob_df = pd.DataFrame(y_pred_proba, columns=[f"P({name})" for name in target_names])
prob_df['True_Label'] = [target_names[i] for i in y_test]
prob_df['Predicted_Label'] = [target_names[i] for i in y_pred]

def predict_origin(features):
    features_scaled = scaler.transform(features)
    pred_classes = nb_model.predict(features_scaled)
    pred_proba = nb_model.predict_proba(features_scaled)

    return pred_classes, pred_proba

print(f"\n--- Detailed statistics of prediction results ---")
comparison_df = pd.DataFrame({
    'Sample_Index': range(len(y_test)),
    'True_Label': [target_names[i] for i in y_test],
    'Predicted_Label': [target_names[i] for i in y_pred],
    'Is_Correct': y_test == y_pred,
    'Confidence': np.max(y_pred_proba, axis=1)
})

print("Prediction result statistics:")
print(comparison_df)

print(f"\nPrediction accuracy by origin:")
for i, class_name in enumerate(target_names):
    class_mask = y_test == i
    if np.any(class_mask):
        class_correct = np.sum((y_test == i) & (y_pred == i))
        class_total = np.sum(y_test == i)
        class_acc = class_correct / class_total
        print(f"{class_name}: {class_correct}/{class_total} = {class_acc:.3f}")

print(f"\n--- Naive Bayes origin identification model construction completed ---")
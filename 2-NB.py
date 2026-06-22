import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.5
})

df = pd.read_excel('Origin–Element–Spectrum.xlsx', header=0)
X = df.iloc[:, 1:].values
y_series = df.iloc[:, 0]

try:
    y = y_series.astype('category').cat.codes.values
    target_names = y_series.astype('category').cat.categories.tolist()
except Exception:
    y = y_series.values
    target_names = ["Unknown"]

print(f"Shape of the raw data: {X.shape}")
print(f"Target class: {target_names}")

X_group1 = X[:, :11]
X_group2 = X[:, 11:]

scaler1 = MinMaxScaler()
scaler2 = StandardScaler()
X_group1_scaled = scaler1.fit_transform(X_group1)
X_group2_scaled = scaler2.fit_transform(X_group2)

def get_pca_components(X, min_variance=0.95):
    pca = PCA()
    pca.fit(X)
    cumsum_ratio = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum_ratio >= min_variance) + 1
    n_components = max(1, n_components)
    print(f"  Target variance: {min_variance:.0%}, Actually retained: {cumsum_ratio[n_components-1]:.4f} ({n_components} components)")
    pca_final = PCA(n_components=n_components)
    X_pca = pca_final.fit_transform(X)
    return X_pca, pca_final

print("\n--- Group 1 PCA ---")
X_group1_pca, pca_group1 = get_pca_components(X_group1_scaled, min_variance=0.95)

print("\n--- Group 2 PCA ---")
X_group2_pca, pca_group2 = get_pca_components(X_group2_scaled, min_variance=0.95)

X_pca_combined = np.hstack([X_group1_pca, X_group2_pca])
print(f"\nTotal dimensionality after fusion: {X_pca_combined.shape[1]} (Elements:{X_group1_pca.shape[1]} + Spectra:{X_group2_pca.shape[1]})")

if X_pca_combined.shape[1] >= 3:
    fig_width_inch = 17.6 / 2.54
    fig_height_inch = 12.0 / 2.54
    fig = plt.figure(figsize=(fig_width_inch, fig_height_inch))
    ax = fig.add_subplot(111, projection='3d')

    fig.patch.set_facecolor('white')
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('k')
        axis.pane.set_alpha(0.1)

    ax.view_init(elev=45, azim=45)

    unique_y = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_y)))
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']

    for i, class_val in enumerate(unique_y):
        mask = y == class_val
        x_data = X_pca_combined[mask, 0]
        y_data = X_pca_combined[mask, 1]
        z_data = X_pca_combined[mask, 2]
        ax.scatter(x_data, y_data, z_data,
                   c=[colors[i]], marker=markers[i % len(markers)],
                   edgecolors='k', s=20, alpha=0.9,
                   label=target_names[class_val])

    ax.set_xlabel(f'PC1 ({pca_group1.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca_group1.explained_variance_ratio_[1]:.2%})')
    ax.set_zlabel(f'PC3 ({pca_group2.explained_variance_ratio_[0]:.2%})')
    ax.legend(title="Class", loc='upper left')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    ax.text(xlim[1], ylim[0] + 0.05, zlim[1] + 0.05, '(a)',
            verticalalignment='top', horizontalalignment='left',
            fontsize=8, fontweight='bold')

    plt.tight_layout(pad=2.0)

    output_png = "Fused_Features_3D_PCA_95var.png"
    plt.savefig(
        output_png,
        format='png',
        dpi=600,
        bbox_inches='tight',
        pad_inches=0.5,
        transparent=False,
        facecolor='white'
    )
    plt.show()

X_train_pca, X_test_pca, y_train, y_test = train_test_split(
    X_pca_combined, y, test_size=0.3, stratify=y, random_state=42
)

nb_model = GaussianNB()
nb_model.fit(X_train_pca, y_train)

y_pred = nb_model.predict(X_test_pca)
train_acc = nb_model.score(X_train_pca, y_train)
test_acc = nb_model.score(X_test_pca, y_test)

print(f"\n--- Model Performance in PCA Space ---")
print(f"Training accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

cm = confusion_matrix(y_test, y_pred)
fig_width_inch = 17.6 / 2.54
fig_height_inch = 12.0 / 2.54
plt.figure(figsize=(fig_width_inch, fig_height_inch))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel('Predicted labels')
plt.ylabel('Actual labels')
plt.text(-0.05, 0.98, '(b)', transform=plt.gca().transAxes,
         verticalalignment='top', horizontalalignment='left',
         fontsize=8, fontweight='bold')
plt.tight_layout(pad=1.0)
plt.savefig(
    'Confusion_Matrix_PCA_95var.png',
    format='png',
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.02,
    transparent=True,
    facecolor='none'
)
plt.show()

print(f"\n--- Classification Report ---")
report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
print(report)

cv_scores = cross_val_score(nb_model, X_pca_combined, y, cv=5, scoring='accuracy')
print(f"\n--- 5-fold cross-validation ---")
print(f"Average Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

X_full_original = np.hstack([X_group1_scaled, X_group2_scaled])
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X_full_original, y, test_size=0.3, stratify=y, random_state=42
)

nb_orig = GaussianNB()
nb_orig.fit(X_train_orig, y_train_orig)
orig_test_acc = nb_orig.score(X_test_orig, y_test_orig)

print(f"\n--- Performance Comparison ---")
print(f"Test Accuracy in Original Feature Space: {orig_test_acc:.4f}")
print(f"Test Accuracy in PCA Space: {test_acc:.4f}")
print(f"Feature Dimensionality: Original {X_full_original.shape[1]} → PCA {X_pca_combined.shape[1]}")
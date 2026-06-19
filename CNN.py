import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from scipy.signal import resample
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# 定义1D-CNN模型
class OneDCNN(nn.Module):
    def __init__(self, input_length=2074, num_classes=6, elu_alpha=1.0, dropout_p=0.5):
        super(OneDCNN, self).__init__()

        # 第一组：卷积 + 最大池化
        # 输入: (batch_size, 1, 2074)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=41)
        # Conv1d输出: (batch_size, 32, 2034)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        # Pool1d输出: (batch_size, 32, 1017)

        # 第二组：卷积 + 最大池化
        # 输入: (batch_size, 32, 1017)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=39)
        # Conv1d输出: (batch_size, 32, 979)
        self.pool2 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)  # ceil_mode=True to match chart (979->490)
        # Pool1d输出: (batch_size, 32, 490)

        # 第三组：卷积 + 最大池化
        # 输入: (batch_size, 32, 490)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=37)
        # Conv1d输出: (batch_size, 32, 454)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        # Pool1d输出: (batch_size, 32, 227)

        # 全连接层
        # 池化后: (batch_size, 32, 227) -> flatten -> (batch_size, 32 * 227 = 7264)
        self.fc1 = nn.Linear(32 * 227, 64)  # 第一个全连接层

        # Dropout 层 - 通常放在全连接层之间或之后
        self.dropout = nn.Dropout(p=dropout_p)  # p 是被置零的概率

        self.fc2 = nn.Linear(64, num_classes)  # 第二个全连接层，输出类别数

        # 激活函数 (ELU) - 可以指定 alpha 参数，默认为 1.0
        self.elu = nn.ELU(alpha=elu_alpha)

    def forward(self, x):
        # x shape: (batch_size, channels=1, length=2074)

        # C1 and S2
        x = self.elu(self.conv1(x))  # -> (batch_size, 32, 2034)
        x = self.pool1(x)  # -> (batch_size, 32, 1017)

        # C3 and S4
        x = self.elu(self.conv2(x))  # -> (batch_size, 32, 979)
        x = self.pool2(x)  # -> (batch_size, 32, 490)

        # C5 and S6
        x = self.elu(self.conv3(x))  # -> (batch_size, 32, 454)
        x = self.pool3(x)  # -> (batch_size, 32, 227)

        # 展平
        x = x.view(x.size(0), -1)  # -> (B, 32 * 227 = 7264)

        # 全连接层
        x = self.elu(self.fc1(x))  # -> (B, 64)

        # 应用 Dropout
        x = self.dropout(x)  # -> (B, 64) - 随机将约 p% 的元素置为 0

        x = self.fc2(x)  # -> (B, 6) - 输出类别logits (未归一化的概率)

        return x  # 返回未经过softmax的logits，CrossEntropyLoss会处理


def standardize_normalize(spectra):
    """
    应用标准化进行光谱预处理（Z-score标准化）

    Parameters:
    spectra: numpy array of shape (n_samples, n_features)

    Returns:
    standardized_spectra: numpy array of shape (n_samples, n_features)
    """
    # 使用sklearn的StandardScaler进行标准化
    scaler = StandardScaler()
    standardized_spectra = scaler.fit_transform(spectra)

    return standardized_spectra

def snv_normalize(spectra):
    """
    应用标准正态变量变换(SNV)进行光谱预处理

    Parameters:
    spectra: numpy array of shape (n_samples, n_features)

    Returns:
    normalized_spectra: numpy array of shape (n_samples, n_features)
    """
    # 计算每个样本的均值和标准差
    mean_spectrum = np.mean(spectra, axis=1, keepdims=True)
    std_spectrum = np.std(spectra, axis=1, keepdims=True)

    # 避免除零错误
    std_spectrum = np.where(std_spectrum == 0, 1, std_spectrum)

    # 应用SNV
    normalized_spectra = (spectra - mean_spectrum) / std_spectrum

    return normalized_spectra


def downsample_spectra(spectra, target_length=2074):
    """
    下采样光谱数据到目标长度

    Parameters:
    spectra: numpy array of shape (n_samples, original_length)
    target_length: int, 目标长度

    Returns:
    downsampled_spectra: numpy array of shape (n_samples, target_length)
    """
    if spectra.shape[1] <= target_length:
        # 如果原始长度小于等于目标长度，直接填充或截断
        if spectra.shape[1] < target_length:
            # 填充零到目标长度
            padded_spectra = np.zeros((spectra.shape[0], target_length))
            padded_spectra[:, :spectra.shape[1]] = spectra
            return padded_spectra
        else:
            # 截断到目标长度
            return spectra[:, :target_length]
    else:
        # 使用scipy的resample进行下采样
        downsampled_spectra = np.zeros((spectra.shape[0], target_length))
        for i in range(spectra.shape[0]):
            downsampled_spectra[i, :] = resample(spectra[i, :], target_length)
        return downsampled_spectra


def preprocess_spectral_data(features, target_length=2074):
    """
    对光谱数据进行预处理：下采样 + SNV

    Parameters:
    features: numpy array, 原始特征数据
    target_length: int, 目标长度

    Returns:
    processed_features: numpy array, 预处理后的特征数据
    """
    print(f"Original data shape: {features.shape}")

    # 步骤1: 下采样到目标长度
    downsampled_features = downsample_spectra(features, target_length)
    print(f"After downsampling: {downsampled_features.shape}")

    # 步骤2: SNV归一化
    snv_normalized_features = StandardScaler().fit_transform(downsampled_features)
    print(f"After SNV normalization: {snv_normalized_features.shape}")

    return snv_normalized_features


def load_excel_data(file_path, label_column='产地', feature_start_col=1):
    """
    从Excel文件加载数据

    Parameters:
    file_path: str, Excel文件路径
    label_column: str, 标签列名
    feature_start_col: int, 特征开始的列索引（从0开始）
    """
    # 读取Excel文件
    df = pd.read_excel(file_path, header=0)

    # 提取特征和标签
    if label_column in df.columns:
        labels = df[label_column].values
        # 提取特征列（除了标签列）
        feature_cols = [col for col in df.columns if col != label_column]
        features = df[feature_cols].values
    else:
        # 如果标签列不存在，假设第一列是标签
        labels = df.iloc[:, 0].values  # 第一列作为标签
        features = df.iloc[:, feature_start_col:].values  # 从指定列开始作为特征

    print(f"Data shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")

    return features, labels


def split_and_prepare_data(features, labels, test_size=0.2, val_size=0.1, random_state=42):
    """
    分割数据并应用预处理（分为训练集、验证集和测试集）

    Parameters:
    features: numpy array, 特征数据
    labels: numpy array, 标签数据
    test_size: float, 测试集比例
    val_size: float, 验证集比例 (相对于剩余数据)
    random_state: int, 随机种子
    """
    # 将标签转换为数值编码
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # 划分训练集和临时集（包含验证集和测试集）
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, encoded_labels, test_size=test_size, random_state=random_state, stratify=encoded_labels
    )

    # 计算验证集在剩余数据中的比例
    remaining_val_size = val_size / (1 - test_size)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=remaining_val_size, random_state=random_state, stratify=y_temp
    )

    # 对所有分割应用预处理
    X_train_processed = preprocess_spectral_data(X_train)
    X_val_processed = preprocess_spectral_data(X_val)
    X_test_processed = preprocess_spectral_data(X_test)

    # 转换为PyTorch张量并调整维度 (添加通道维度)
    X_train_tensor = torch.FloatTensor(X_train_processed).unsqueeze(1)  # (batch_size, 1, 2074)
    X_val_tensor = torch.FloatTensor(X_val_processed).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test_processed).unsqueeze(1)

    y_train_tensor = torch.LongTensor(y_train)
    y_val_tensor = torch.LongTensor(y_val)
    y_test_tensor = torch.LongTensor(y_test)

    return {
        'train': (X_train_tensor, y_train_tensor),
        'val': (X_val_tensor, y_val_tensor),
        'test': (X_test_tensor, y_test_tensor),
        'label_encoder': label_encoder
    }


def create_dataloaders(processed_data, batch_size=32, shuffle=True):
    """
    创建数据加载器

    Parameters:
    processed_ dict, 预处理后的数据字典
    batch_size: int, 批次大小
    shuffle: bool, 是否打乱数据
    """
    train_dataset = TensorDataset(processed_data['train'][0], processed_data['train'][1])
    val_dataset = TensorDataset(processed_data['val'][0], processed_data['val'][1])
    test_dataset = TensorDataset(processed_data['test'][0], processed_data['test'][1])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def validate_model(model, val_loader, device):
    """
    在验证集上评估模型
    """
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions / total_samples
    return accuracy


def train_model_with_validation(model, train_loader, val_loader, device, num_epochs=50, learning_rate=0.001,
                                patience=30):
    """
    使用验证集训练模型，并保存最佳模型
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    print("Starting training with validation...")

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct_predictions / total_samples

        # 验证阶段
        val_acc = validate_model(model, val_loader, device)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_state = model.state_dict().copy()  # 保存最佳模型状态
            epochs_without_improvement = 0
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} (Best!)')
        else:
            epochs_without_improvement += 1
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

        # 早停机制 - 增加耐心值
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Training finished. Best validation accuracy: {best_val_accuracy:.4f}')
    else:
        print('Training finished.')

    return model


def evaluate_model(model, test_loader, device, label_encoder):
    """
    评估模型并生成混淆矩阵
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 转换回原始标签名称
    original_labels = label_encoder.inverse_transform(all_labels)
    original_preds = label_encoder.inverse_transform(all_preds)

    # 计算准确率
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f'Test Accuracy: {accuracy:.4f}')

    # 生成混淆矩阵
    cm = confusion_matrix(original_labels, original_preds)

    # 打印分类报告（解决 precision warning）
    print("\nClassification Report:")
    print(classification_report(original_labels, original_preds, zero_division=0))

    # === 设置 Food Control 合规样式（全英文 + Arial）===
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'pdf.fonttype': 42,  # 嵌入字体
        'ps.fonttype': 42,
        'axes.linewidth': 0.5
    })

    # 创建图形（双栏宽度 17.6 cm）
    fig_width_inch = 17.6 / 2.54  # = 6.93 inch
    fig_height_inch = 12.0 / 2.54  # 高度 ~12 cm
    plt.figure(figsize=(fig_width_inch, fig_height_inch))

    unique_labels = label_encoder.classes_

    # 绘制混淆矩阵
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=unique_labels,
                yticklabels=unique_labels,
                cbar_kws={'shrink': 0.8})  # 调整色条大小

    # 坐标轴（全英文）
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(rotation=45, ha='right')  # 右对齐避免重叠
    plt.yticks(rotation=0)

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

    plt.show()  # 关闭图形以释放内存

    return cm, accuracy


def load_and_prepare_excel_data(file_path, label_column='产地', feature_start_col=1, batch_size=32, test_size=0.2,
                                val_size=0.1):
    """
    加载和准备Excel数据的主函数（包含下采样+SNV预处理，分为训练集、验证集和测试集）

    Parameters:
    file_path: str, Excel文件路径
    label_column: str, 标签列名
    feature_start_col: int, 特征开始的列索引
    batch_size: int, 批次大小
    test_size: float, 测试集比例
    val_size: float, 验证集比例 (相对于剩余数据)
    """
    # 加载数据
    features, labels = load_excel_data(file_path, label_column, feature_start_col)

    # 分割数据并应用预处理（下采样+SNV）
    processed_data = split_and_prepare_data(features, labels, test_size=test_size, val_size=val_size)

    # 创建数据加载器
    dataloaders = create_dataloaders(processed_data, batch_size=batch_size)

    return dataloaders, processed_data


def main():
    # 加载Excel数据并应用下采样+SNV预处理
    print("Loading data...")
    dataloaders, processed_data = load_and_prepare_excel_data(
        file_path='核心-光谱.xlsx',
        label_column='产地',  # 标签列名
        feature_start_col=1,  # 特征从第2列开始（索引为1）
        batch_size=32,
        test_size=0.2,  # 80%用于训练+验证，20%用于测试
        val_size=0.125  # 在剩余80%中，10%用于验证，70%用于训练
    )

    # 获取模型和设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = OneDCNN(input_length=2074, num_classes=len(np.unique(processed_data['train'][1].numpy())),
                    elu_alpha=1.0, dropout_p=0.5).to(device)

    # 训练模型（使用验证集选择最佳模型）
    print("\nTraining model with validation...")
    model = train_model_with_validation(
        model,
        dataloaders['train'],
        dataloaders['val'],
        device,
        num_epochs=100,  # 增加训练轮次
        learning_rate=0.001,
        patience=1000  # 增加耐心值，防止过早停止
    )

    # 评估最佳模型并输出混淆矩阵
    print("\nEvaluating best model on test set...")
    cm, accuracy = evaluate_model(model, dataloaders['test'], device, processed_data['label_encoder'])

    print(f"\nFinal Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
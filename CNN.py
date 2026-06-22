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

class OneDCNN(nn.Module):
    def __init__(self, input_length=2074, num_classes=6, elu_alpha=1.0, dropout_p=0.5):
        super(OneDCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=41)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=39)
        self.pool2 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=37)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 227, 64)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(64, num_classes)
        self.elu = nn.ELU(alpha=elu_alpha)

    def forward(self, x):
        x = self.elu(self.conv1(x))
        x = self.pool1(x)
        x = self.elu(self.conv2(x))
        x = self.pool2(x)
        x = self.elu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def standardize_normalize(spectra):
    scaler = StandardScaler()
    standardized_spectra = scaler.fit_transform(spectra)

    return standardized_spectra

def snv_normalize(spectra):
    mean_spectrum = np.mean(spectra, axis=1, keepdims=True)
    std_spectrum = np.std(spectra, axis=1, keepdims=True)

    std_spectrum = np.where(std_spectrum == 0, 1, std_spectrum)

    normalized_spectra = (spectra - mean_spectrum) / std_spectrum

    return normalized_spectra


def downsample_spectra(spectra, target_length=2074):
    if spectra.shape[1] <= target_length:
        if spectra.shape[1] < target_length:
            padded_spectra = np.zeros((spectra.shape[0], target_length))
            padded_spectra[:, :spectra.shape[1]] = spectra
            return padded_spectra
        else:
            return spectra[:, :target_length]
    else:
        downsampled_spectra = np.zeros((spectra.shape[0], target_length))
        for i in range(spectra.shape[0]):
            downsampled_spectra[i, :] = resample(spectra[i, :], target_length)
        return downsampled_spectra


def preprocess_spectral_data(features, target_length=2074):
    print(f"Original data shape: {features.shape}")

    downsampled_features = downsample_spectra(features, target_length)
    print(f"After downsampling: {downsampled_features.shape}")

    snv_normalized_features = StandardScaler().fit_transform(downsampled_features)
    print(f"After SNV normalization: {snv_normalized_features.shape}")

    return snv_normalized_features


def load_excel_data(file_path, label_column='Origin', feature_start_col=1):
    df = pd.read_excel(file_path, header=0)

    if label_column in df.columns:
        labels = df[label_column].values
        feature_cols = [col for col in df.columns if col != label_column]
        features = df[feature_cols].values
    else:
        labels = df.iloc[:, 0].values
        features = df.iloc[:, feature_start_col:].values

    print(f"Data shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")

    return features, labels


def split_and_prepare_data(features, labels, test_size=0.2, val_size=0.1, random_state=42):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    X_temp, X_test, y_temp, y_test = train_test_split(
        features, encoded_labels, test_size=test_size, random_state=random_state, stratify=encoded_labels
    )

    remaining_val_size = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=remaining_val_size, random_state=random_state, stratify=y_temp
    )

    X_train_processed = preprocess_spectral_data(X_train)
    X_val_processed = preprocess_spectral_data(X_val)
    X_test_processed = preprocess_spectral_data(X_test)

    X_train_tensor = torch.FloatTensor(X_train_processed).unsqueeze(1)
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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    print("Starting training with validation...")

    for epoch in range(num_epochs):
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

        val_acc = validate_model(model, val_loader, device)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} (Best!)')
        else:
            epochs_without_improvement += 1
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Training finished. Best validation accuracy: {best_val_accuracy:.4f}')
    else:
        print('Training finished.')

    return model


def evaluate_model(model, test_loader, device, label_encoder):
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

    original_labels = label_encoder.inverse_transform(all_labels)
    original_preds = label_encoder.inverse_transform(all_preds)

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f'Test Accuracy: {accuracy:.4f}')

    cm = confusion_matrix(original_labels, original_preds)

    print("\nClassification Report:")
    print(classification_report(original_labels, original_preds, zero_division=0))

    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.linewidth': 0.5
    })

    fig_width_inch = 17.6 / 2.54
    fig_height_inch = 12.0 / 2.54
    plt.figure(figsize=(fig_width_inch, fig_height_inch))

    unique_labels = label_encoder.classes_

    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=unique_labels,
                yticklabels=unique_labels,
                cbar_kws={'shrink': 0.8})

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

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

    return cm, accuracy


def load_and_prepare_excel_data(file_path, label_column='Origin', feature_start_col=1, batch_size=32, test_size=0.2,
                                val_size=0.1):
    features, labels = load_excel_data(file_path, label_column, feature_start_col)

    processed_data = split_and_prepare_data(features, labels, test_size=test_size, val_size=val_size)

    dataloaders = create_dataloaders(processed_data, batch_size=batch_size)

    return dataloaders, processed_data


def main():
    print("Loading data...")
    dataloaders, processed_data = load_and_prepare_excel_data(
        file_path='Core–Spectrum.xlsx',
        label_column='Origin',
        feature_start_col=1,
        batch_size=32,
        test_size=0.2,
        val_size=0.125
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = OneDCNN(input_length=2074, num_classes=len(np.unique(processed_data['train'][1].numpy())),
                    elu_alpha=1.0, dropout_p=0.5).to(device)

    print("\nTraining model with validation...")
    model = train_model_with_validation(
        model,
        dataloaders['train'],
        dataloaders['val'],
        device,
        num_epochs=100,
        learning_rate=0.001,
        patience=1000
    )

    print("\nEvaluating best model on test set...")
    cm, accuracy = evaluate_model(model, dataloaders['test'], device, processed_data['label_encoder'])

    print(f"\nFinal Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
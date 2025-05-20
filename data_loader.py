import torch
import numpy as np
import os
from torchvision import datasets, transforms

def load_standard_mnist():
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    X_train = torch.stack([img for img, _ in dataset])
    y_train = torch.tensor([label for _, label in dataset])
    X_test = torch.stack([img for img, _ in testset])
    y_test = torch.tensor([label for _, label in testset])
    return X_train, y_train, X_test, y_test

def select_severity(X, y, severity, total_severities=5):
    n = len(X) // total_severities
    start = n * (severity - 1)
    end = n * severity
    return X[start:end], y[start:end]

def load_mnist_c(corruption_folder, severity=3):
    base_path = os.path.join("mnist_c", corruption_folder)

    try:
        X_train_raw = np.load(os.path.join(base_path, "train_images.npy")) / 255.0
        y_train_raw = np.load(os.path.join(base_path, "train_labels.npy"))
        X_test_raw = np.load(os.path.join(base_path, "test_images.npy")) / 255.0
        y_test_raw = np.load(os.path.join(base_path, "test_labels.npy"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivos não encontrados ou com nomes incorretos em {base_path}")

    # Fatiamento por severidade (mais robusto)
    X_train, y_train = select_severity(X_train_raw, y_train_raw, severity)
    X_test, y_test = select_severity(X_test_raw, y_test_raw, severity)

    # Ajuste de dimensão: [N, H, W, 1] → [N, 1, H, W]
    X_train = torch.tensor(X_train).float()
    if X_train.ndim == 4 and X_train.shape[-1] == 1:
        X_train = X_train.permute(0, 3, 1, 2)

    X_test = torch.tensor(X_test).float()
    if X_test.ndim == 4 and X_test.shape[-1] == 1:
        X_test = X_test.permute(0, 3, 1, 2)

    y_train = torch.tensor(y_train).long()
    y_test = torch.tensor(y_test).long()

    return X_train, y_train, X_test, y_test


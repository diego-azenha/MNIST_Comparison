import torch
from data_loader import load_standard_mnist, load_mnist_c

import numpy as np

from ConvNN import train as train_cnn, predict as predict_cnn
from MLP import train as train_mlp, predict as predict_mlp
from RandomForest import train as train_rf, predict as predict_rf
from metrics import plot_confusion_matrix, plot_roc_ova, detailed_metrics

import os

# Cria pasta de resultados se não existir
os.makedirs("results", exist_ok=True)

# 1. Pré-processamento
print("Escolha o dataset:")
print("1 - MNIST padrão")
print("2 - MNIST-C (corrompido)")
model_tag = ""

choice = input("Digite 1 ou 2: ")

if choice == "1":
    print("Carregando MNIST padrão...")
    X_train, y_train, X_test, y_test = load_standard_mnist()
    model_tag = "MNIST"
elif choice == "2":
    print("\nCorrupções disponíveis em MNIST-C:")
    corruptions = sorted(os.listdir("mnist_c"))
    for i, name in enumerate(corruptions):
        print(f"{i + 1} - {name}")
    idx = int(input("Escolha a corrupção (número): ")) - 1
    corruption = corruptions[idx]

    severity = int(input("Escolha a severidade (1 a 5): "))
    print(f"Carregando MNIST-C ({corruption}, severidade {severity})...")
    X_train, y_train, X_test, y_test = load_mnist_c(corruption, severity)
    model_tag = f"MNISTC_{corruption}_s{severity}"
else:
    raise ValueError("Escolha inválida.")


# Também como NumPy para o Random Forest
X_train_np = X_train.numpy()
y_train_np = y_train.numpy()
X_test_np = X_test.numpy()
y_test_np = y_test.numpy()

print("Pré-processamento concluído")

# 2. CNN
print("="*40)
print("Iniciando treinamento da CNN")
train_cnn(X_train, y_train, epochs=5, batch_size=64, learning_rate=0.001)
print("Predição com CNN...")
print("Shape de X_test:", X_test.shape)
y_pred_cnn, y_proba_cnn = predict_cnn(X_test)
print("Gerando gráficos da CNN...")
plot_confusion_matrix(y_test_np, y_pred_cnn, f"CNN_{model_tag}")
plot_roc_ova(y_test_np, y_proba_cnn, f"CNN_{model_tag}")


# 3. MLP
print("="*40)
print("Iniciando treinamento da MLP")
train_mlp(X_train, y_train, epochs=5, batch_size=64, learning_rate=0.001)
print("Predição com MLP...")
y_pred_mlp, y_proba_mlp = predict_mlp(X_test)
print("Gerando gráficos da MLP...")
plot_confusion_matrix(y_test_np, y_pred_mlp, f"MLP_{model_tag}")
plot_roc_ova(y_test_np, y_proba_mlp, f"MLP_{model_tag}")


# 4. Random Forest
print("="*40)
print("Iniciando treinamento da Random Forest")
train_rf(X_train_np, y_train_np, n_estimators=100)
print("Predição com Random Forest...")
y_pred_rf, y_proba_rf = predict_rf(X_test_np)
print("Gerando gráficos da Random Forest...")
plot_confusion_matrix(y_test_np, y_pred_rf, f"RF_{model_tag}")
plot_roc_ova(y_test_np, y_proba_rf, f"RF_{model_tag}")


# Detailed metrics
detailed_metrics(y_test_np, y_pred_cnn, y_proba_cnn, f"CNN_{model_tag}")
detailed_metrics(y_test_np, y_pred_mlp, y_proba_mlp, f"MLP_{model_tag}")
detailed_metrics(y_test_np, y_pred_rf, y_proba_rf, f"RF_{model_tag}")
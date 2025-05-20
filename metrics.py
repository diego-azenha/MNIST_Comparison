import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression  # usado apenas para compatibilidade da API ROC

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.savefig(f"results/{model_name}_confusion_matrix.png")
    plt.close()

def plot_roc_ova(y_true, y_proba, model_name):
    n_classes = y_proba.shape[1]
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Classe {i} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de Falso Positivo")
    plt.ylabel("Taxa de Verdadeiro Positivo")
    plt.title(f"Curvas ROC One-vs-All - {model_name}")
    plt.legend(loc="lower right")
    plt.savefig(f"results/{model_name}_roc_ova.png")
    plt.close()

from sklearn.metrics import classification_report, log_loss, f1_score

def detailed_metrics(y_true, y_pred, y_proba, model_name):
    print("="*40)
    print(f"🔍 DETAILED METRICS - {model_name}")
    print("="*40)

    # 1. F1-score por classe (e matriz implícita)
    print("\n📌 Classification Report (Precision, Recall, F1 por classe):")
    print(classification_report(y_true, y_pred, digits=3))

    # 2. Log loss
    loss = log_loss(y_true, y_proba)
    print(f"\n🧮 Log Loss (Cross-Entropy): {loss:.4f}")

    # 3. Macro F1-score
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"📊 Macro F1-Score: {macro_f1:.4f}\n")

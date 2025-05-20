import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from tqdm import tqdm

MODEL_PATH = "modelos/rf_model.pkl"
SCALER_PATH = "modelos/rf_scaler.pkl"

def train(X_train, y_train, n_estimators=100, max_depth=None):
    X_train_flat = X_train.reshape(X_train.shape[0], -1)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)

    # Configura para treinamento incremental
    clf = RandomForestClassifier(
        n_estimators=1,
        max_depth=max_depth,
        warm_start=True,
        random_state=42
    )

    print(f"Iniciando treinamento da Random Forest com {n_estimators} árvores...")

    for i in tqdm(range(1, n_estimators + 1), desc="Árvores treinadas", leave=True):
        clf.n_estimators = i
        clf.fit(X_train_scaled, y_train)

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Treinamento concluído. Modelo salvo.")
    return clf

def predict(X_test):
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    X_test_scaled = scaler.transform(X_test_flat)

    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)
    return y_pred, y_proba

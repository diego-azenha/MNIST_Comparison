import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm

MODEL_PATH = "modelos/cnn_model.pt"

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 → 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                            # 28x28 → 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14 → 14x14
            nn.ReLU(),
            nn.MaxPool2d(2)                             # 14x14 → 7x7
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

def train(X_train, y_train, batch_size=64, epochs=5, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for i, (batch_X, batch_y) in enumerate(progress_bar):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)

            progress_bar.set_postfix(batch=i+1, loss=avg_loss)


    torch.save(model.state_dict(), MODEL_PATH)
    return model

def predict(X_test, batch_size=64):
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Certifica-se de que X_test está em TensorDataset
    dataloader = DataLoader(TensorDataset(X_test), batch_size=batch_size)

    all_preds = []
    all_probas = []

    with torch.no_grad():
        for batch in dataloader:
            batch_X = batch[0].to(device)  # extrai o tensor do tuple
            outputs = model(batch_X)
            probas = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probas, dim=1)

            all_preds.append(preds.cpu())
            all_probas.append(probas.cpu())

    # Verificação de segurança: evitar erro se dataloader não entregou nada
    if not all_preds:
        raise RuntimeError("Nenhum batch foi processado — verifique o formato de X_test.")

    y_pred = torch.cat(all_preds).numpy()
    y_proba = torch.cat(all_probas).numpy()
    return y_pred, y_proba


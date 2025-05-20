import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

MODEL_PATH = "modelos/mlp_model.pt"

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),                   # 1x28x28 → 784
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

def train(X_train, y_train, batch_size=64, epochs=5, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    dataloader = DataLoader(TensorDataset(X_test), batch_size=batch_size)
    all_preds = []
    all_probas = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch[0].to(device)  # extraindo o tensor de entrada
            outputs = model(batch)
            probas = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probas, dim=1)

            all_preds.append(preds.cpu())
            all_probas.append(probas.cpu())


    y_pred = torch.cat(all_preds).numpy()
    y_proba = torch.cat(all_probas).numpy()
    return y_pred, y_proba

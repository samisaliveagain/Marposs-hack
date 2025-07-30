import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------
# Dataset wrapper
# ---------------------------
class CNCDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------
# 1D CNN model
# ---------------------------
class CNNClassifier(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(64 * 19, 100),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, n_classes)
        )

    def forward(self, x):
        return self.model(x)

# ---------------------------
# Load data
# ---------------------------
X_train = np.load("split_cnc_data/X_train.npy")
y_train = np.load("split_cnc_data/y_train.npy")
X_test = np.load("split_cnc_data/X_test.npy")
y_test = np.load("split_cnc_data/y_test.npy")

train_dataset = CNCDataset(X_train, y_train)
test_dataset = CNCDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ---------------------------
# Training
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier(in_channels=X_train.shape[1], n_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 21):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch}: Loss = {running_loss / len(train_loader):.4f}")

# ---------------------------
# Evaluation
# ---------------------------
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

print("\nTest Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=["NIO", "IO"]))

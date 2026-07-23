# Standalone smoke test of train_m3.ipynb's logic (no Colab/Drive dependency),
# run against the real Phase 3 M3 dataset to verify the pipeline end-to-end.
import pandas as pd, numpy as np, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

DATA = "/home/claude/Mentora/datasets"
MODELS = "/home/claude/Mentora/training/smoke_test/models"
os.makedirs(f"{MODELS}/model3_trajectory_predictor", exist_ok=True)

df = pd.read_csv(f"{DATA}/processed/m3/score_histories_no_label.csv")

SEQ_LEN = 5
class TrajectoryDataset(Dataset):
    def __init__(self, df, seq_len=SEQ_LEN):
        self.samples = []
        for student_id, group in df.groupby("student_id"):
            scores = group.sort_values("session_number")["score"].values
            for i in range(len(scores) - seq_len):
                x = scores[i:i+seq_len]
                y = scores[i+seq_len]
                self.samples.append((x, y))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)

full_ds = TrajectoryDataset(df)
n_val = int(0.15 * len(full_ds))
torch.manual_seed(42)
train_ds, val_ds = torch.utils.data.random_split(full_ds, [len(full_ds) - n_val, n_val])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
print(f"{len(train_ds)} train windows, {len(val_ds)} val windows")

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TrajectoryLSTM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

N_EPOCHS = 30
val_mae = None
for epoch in range(N_EPOCHS):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
    train_loss /= len(train_ds)

    model.eval()
    val_abs_errors = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            val_abs_errors.extend((pred - y).abs().cpu().tolist())
    val_mae = np.mean(val_abs_errors)

    print(f"epoch {epoch+1}/{N_EPOCHS} - train_loss {train_loss:.3f} - val_MAE {val_mae:.3f}")

    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch},
               f"{MODELS}/model3_trajectory_predictor/checkpoint.pt")

    if val_mae <= 5.0:
        print(f"Target MAE <= 5 reached at epoch {epoch+1} (val_MAE={val_mae:.2f}) - stopping")
        break

torch.save(model.state_dict(), f"{MODELS}/model3_trajectory_predictor/lstm_final.pt")
print("M3 smoke test done. Target: MAE <= 5. Achieved:", val_mae)

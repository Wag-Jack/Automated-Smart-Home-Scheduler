import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and preprocess the dataset
df = pd.read_csv("zigbee0_zgb_packets_db.csv")
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
df['hour'] = df['datetime'].dt.floor('H')

# Aggregate features per hour
features = df.groupby('hour').agg(
    packet_count=('timestamp', 'count'),
    avg_packet_length=('packet_length', 'mean'),
    avg_data_length=('data_length', 'mean'),
    unique_src_addrs=('src_zb_addr', 'nunique'),
    unique_dst_addrs=('dst_zb_addr', 'nunique')
).reset_index()

# Label: 1 if packet_count > median, else 0
median_packets = features['packet_count'].median()
features['activity'] = (features['packet_count'] > median_packets).astype(int)

# Prepare features and labels
X = features.drop(columns=['hour', 'activity']).values
y = features['activity'].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Define model
class ActivityClassifier(nn.Module):
    def __init__(self, input_size):
        super(ActivityClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = ActivityClassifier(X.shape[1])

# Training setup
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(20):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# Predict schedule on full dataset
model.eval()
with torch.no_grad():
    predictions = model(X_tensor).squeeze().round().numpy().astype(int)

# Save the predicted schedule
schedule = features[['hour']].copy()
schedule['predicted_activity'] = predictions
schedule.to_csv("predicted_schedule.csv", index=False)
print("Predicted schedule saved to 'predicted_schedule.csv'")

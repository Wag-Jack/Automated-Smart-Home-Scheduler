import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

# Load dataset
file_path = "zigbee0_zgb_packets_db.csv"
df = pd.read_csv(file_path)

# Convert timestamp to datetime
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek

# Encode 'data' column
encoder = LabelEncoder()
df['data_encoded'] = encoder.fit_transform(df['data'])


# Define activity states
def categorize_activity(hour):
    if 0 <= hour < 6:
        return 0  # Asleep
    elif 6 <= hour < 18:
        return 1  # Active
    else:
        return 2  # Out of house


df['activity'] = df['hour'].apply(categorize_activity)

# Select features and target
features = ['hour', 'day_of_week', 'data_encoded']
target = 'activity'
X = df[features]
y = df[target]

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# PyTorch Dataset class
class HomeActivityDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = HomeActivityDataset(X_train, y_train)
test_dataset = HomeActivityDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define PyTorch model
class ActivityModel(nn.Module):
    def __init__(self, input_size):
        super(ActivityModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3)  # 3 activity classes

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Model initialization
model = ActivityModel(input_size=X.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

# Save model
torch.save(model.state_dict(), "home_activity_model.pth")
print("Model training complete and saved.")

# Load the trained model
model = ActivityModel(input_size=3)  # Ensure the input size matches training
model.load_state_dict(torch.load("home_activity_model.pth"))
model.eval()

# Initialize the scaler (use the same one from training)
scaler = StandardScaler()
scaler.fit(X)  # Use the original training data's X to fit the scaler

# Generate a week's worth of timestamps
start_time = datetime.now()
schedule = []
for i in range(7 * 24):  # 7 days * 24 hours
    timestamp = start_time + timedelta(hours=i)
    hour = timestamp.hour
    day_of_week = timestamp.weekday()

    # Example 'data_encoded' value (adjust based on real data patterns)
    data_encoded = 0  # Assume the most common data pattern

    # Prepare input
    input_features = scaler.transform([[hour, day_of_week, data_encoded]])
    input_tensor = torch.tensor(input_features, dtype=torch.float32)

    # Predict activity
    with torch.no_grad():
        prediction = model(input_tensor)
        predicted_label = torch.argmax(prediction, dim=1).item()

    # Map prediction to human-readable activity
    activity_map = {0: "Asleep", 1: "Active", 2: "Out of House"}
    schedule.append((timestamp.strftime("%Y-%m-%d %H:%M"), activity_map[predicted_label]))

# Convert to DataFrame and display schedule
schedule_df = pd.DataFrame(schedule, columns=["Timestamp", "Predicted Activity"])
print(schedule_df)

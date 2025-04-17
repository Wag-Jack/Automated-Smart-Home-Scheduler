import google.generativeai as genai
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import io
import paho.mqtt.client as mqtt
import os
import time

genai.configure(api_key="AIzaSyBXZkHQrw8gUi0gj-CcOvtpCqjgMVG6LKk")
model = genai.GenerativeModel("gemini-2.0-flash-lite")

#Broker attributes 
sub_topic = 'zigbee2mqtt/device_id/temperature' #HASS Core -> ML Model
pub_topic = 'scheduler/update'                  #ML Model -> Reasoning Agent
network = '192.168.0.101'
port = 1883
keepalive = 0

def define_schedule(file_name):
    #"zigbee0_zgb_packets_db.csv"

    # Load and preprocess the dataset
    df = pd.read_csv(file_name)
    
    print("CSV Columns:", df.columns)  # Debug which columns exist
    
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

def llm_agent(prompt: str):
    response = model.generate_content(prompt)
    return response.text

def load_schedule(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def summarize_schedule(df: pd.DataFrame) -> str:
    summary = "".join([
        f"At {row['hour']}:00, activity is {row['predicted_activity']}\n"
        for _, row in df.iterrows()
    ])
    return summary

def refine_schedule(file_name):
    schedule_df = load_schedule(file_name)
    schedule_summary = summarize_schedule(schedule_df)

    prompt = (
        "You are a smart home assistant. Based on the user's predicted activity schedule, "
        "suggest a consistent weekly schedule for activating or deactivating a temperature sensor. Only provide went to activate and deactivate the sensor.\n\n"
        "Each row in your response must include the following fields, comma-separated:\n"
        "MONTH, DAY, YEAR, HOUR, MINUTE, SECOND, SENSOR, SENSOR_STATE\n"
        "- Use the current year (2025) for all entries.\n"
        "- Use MONTH=4 (April), and assign each DAY from 14 to 20 to represent Monday to Sunday.\n"
        "- Use MINUTE=0 and SECOND=0.\n"
        "- SENSOR must always be 'temperature sensor'.\n"
        "- SENSOR_STATE must be either 'ON' or 'OFF'.\n"
        "Only output actual schedule rows—do not include any explanation or formatting beyond the schedule.\n\n"
        "Here is the predicted activity schedule:\n" + schedule_summary
    )

    response = llm_agent(prompt)
    print("Sensor Schedule:\n", response)

def process_message(msg):
    csv_payload = msg.payload.decode('utf-8')
    f = io.StringIO(csv_payload)
    define_schedule(f)
    refine_schedule('predicted_schedule.csv')

def publish_schedule(client):
    if not os.path.isfile('./predicted_schedule.csv'):
        print(f'Schedule was not saved to directory, please examine.')
        return
    
    with open('./predicted_schedule.csv', 'rb') as f:
        schedule_data = f.read()

    client.publish(sub_topic, schedule_data)
    print('Initial schedule sent to reasoning agent')

#Callback messages for broker
def on_connect(client, data, flags, rc, properties=None):
    if rc == 0:
        print(f'Scheduler ML connected with result code {rc}')
        client.subscribe(sub_topic)
    else:
        print(f'Failed to connect, result code {rc}')

def on_disconnect(client, data, flags, rc, properties=None):
    print(f'Scheduler ML disconnected with result code {rc}')

def on_message(client, data, message):
    print(f'Message received: {message.payload.decode} on topic {message.topic}')
    process_message(message)

    #Once message processed and schedule made, publish it to agent
    publish_schedule(client)

#New MQTT client
broker = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

#Callback functions
broker.on_connect = on_connect
broker.on_disconnect = on_disconnect
broker.on_message = on_message

#Loop to ensure docker container does not exit early / constant reboot
while True:
    try:
        broker.connect(network, port, keepalive)
        broker.loop_start()

        #Keeps main thread active
        while True:
            time.sleep(1)

    except Exception as e:
        print(f'Connection interrupted, retrying (Reason: {e}')
    
    finally:
        broker.loop_stop()
        broker.disconnect()

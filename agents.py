import google.generativeai as genai
import pandas as pd
import paho.mqtt.client as mqtt
import io
import os
import time

genai.configure(api_key="INSERT KEY HERE")
model = genai.GenerativeModel("gemini-2.0-flash-lite")

#Broker attributes
sub_topic = 'scheduler/initial'
pub_topic = 'scheduler/final'
broker = 'mqtt_broker'
port = 1883
keepalive = 0

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
    #"predicted_schedule.csv"

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
    refine_schedule(f)

def publish_schedule(client):
    if not os.path.isfile('predicted_schedule.csv'):
        print(f'Refined schedule was not saved to directory, please examine.')
        return
    
    with open('predicted_schedule.csv', 'rb') as f:
        schedule_data = f.read()

    client.publish(sub_topic, schedule_data)
    print('Finalized schedule sent to Home Assistant')

#Callback messages for broker
def on_connect(client, data, flags, rc):
    if rc == 0:
        print(f'Reasoning agent connected with result code {rc}')
        client.subscribe(sub_topic)
    else:
        print(f'Failed to connect, result code {rc}')

def on_disconnect(client, data, rc):
    print(f'Reasoning agent disconnected with result code {rc}')

def on_message(client, data, message):
    print(f'Message received: {message.payload.decode} on topic {message.topic}')
    process_message(message)

    #Once message processed and schedule refined, publish it to Home Assistant
    publish_schedule(client)

#New MQTT client
broker = mqtt.Client()

#Callback functions
broker.on_connect = on_connect
broker.on_disconnect = on_disconnect
broker.on_message = on_message

#Loop to ensure docker container does not exit early / constant reboot
while True:
    try:
        broker.connect(broker, port, keepalive)
        broker.loop_start()

        #Keeps main thread active
        while True:
            time.sleep(1)

    except Exception as e:
        print('Connection on reasoning agent interrupted, retrying')

    finally:
        broker.loop_stop()
        broker.disconnect()
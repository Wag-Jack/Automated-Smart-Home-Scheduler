import google.generativeai as genai
import pandas as pd
import paho.mqtt.client as mqtt

genai.configure(api_key="API_KEY_HERE")
model = genai.GenerativeModel("gemini-2.0-flash-lite")

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "home/sensor/agent"

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

def publish_mqtt(message: str):
    client = mqtt.Client()
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.publish(MQTT_TOPIC, message)
    client.disconnect()

if __name__ == "__main__":
    schedule_df = load_schedule("predicted_schedule.csv")
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
    
    with open("gemini_sensor_schedule.txt", "w") as f:
        f.write(response)

    publish_mqtt(response)
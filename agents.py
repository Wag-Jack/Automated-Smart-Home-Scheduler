import google.generativeai as genai
import pandas as pd
import paho.mqtt.client as mqtt

genai.configure(api_key="AIzaSyBXZkHQrw8gUi0gj-CcOvtpCqjgMVG6LKk")
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
        "Given the following activity schedule,\
              suggest a weekly schedule for when to \
            activate and deactivate a temperature\
                  sensor for Home Assitant. No explanation needed.\n\n"
        + schedule_summary
    )

    response = llm_agent(prompt)
    print("Sensor Schedule:\n", response)
    publish_mqtt(response)
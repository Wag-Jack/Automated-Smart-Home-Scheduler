import google.generativeai as genai
import pandas as pd

genai.configure(api_key="AIzaSyBXZkHQrw8gUi0gj-CcOvtpCqjgMVG6LKk")

model = genai.GenerativeModel("gemini-2.0-flash-lite")

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

if __name__ == "__main__":
    schedule_df = load_schedule("predicted_schedule.csv")
    schedule_summary = summarize_schedule(schedule_df)

    prompt = (
        "Given the following activity schedule, suggest when to activate or deactivate a temperature sensor. Be specific with the times and ensure there is a buffer period so that the desired temperature is reached on time.\n\n"
        + schedule_summary
    )

    response = llm_agent(prompt)
    print("Sensor Schedule:\n", response)
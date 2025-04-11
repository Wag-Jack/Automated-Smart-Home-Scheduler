import pandas as pd
from langchain_core.messages import HumanMessage
from agents import scheduler_agent  # Make sure agents.py is in the same directory

# Load the predicted schedule
df = pd.read_csv("predicted_schedule.csv")

# Format the activity data into a human-readable schedule
schedule_summary = ""
for _, row in df.iterrows():
    time_str = row['hour']
    activity = "active" if row['predicted_activity'] == 1 else "inactive"
    schedule_summary += f"At {time_str}, the user is {activity}.\n"

# Prompt for the LLM agent
prompt = f"""
You are a smart home scheduling assistant.

Below is the predicted user activity over time. Based on this, generate a recommended schedule for turning ON and OFF smart home utilities such as lighting, HVAC, and appliances.

Only turn utilities ON during active periods and OFF during inactive ones, unless you have a compelling reason to do otherwise. Be clear and list specific times.

User Activity Schedule:
{schedule_summary}
"""

# Ask the agent for a schedule recommendation
print("Generating smart home utility schedule...\n")
for step, metadata in scheduler_agent.stream({"messages": [HumanMessage(content=prompt)]}, stream_mode="messages"):
    if metadata['langgraph_node'] == 'agent' and (text := step.text()):
        print(text, end='')

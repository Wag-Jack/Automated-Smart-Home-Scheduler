import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI  # ✅ Corrected import
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Load the API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("Missing OPENAI_API_KEY in your environment variables.")

# Initialize model (use model name 'gpt-4o' or 'gpt-4', etc.)
model = ChatOpenAI(model="gpt-4o", api_key=api_key)

# Tools can be added later (e.g., smart home control tools)
tools = []

# Create your agent
scheduler_agent = create_react_agent(model=model, tools=tools)

# Optional interactive mode for direct testing
if __name__ == "__main__":
    print("Smart Home Scheduler Agent is running...\n")
    while True:
        user_input = input("🧠 > ")
        if user_input.lower() in ["exit", "quit"]:
            break
        for step, metadata in scheduler_agent.stream({"messages": [HumanMessage(content=user_input)]}, stream_mode="messages"):
            if metadata['langgraph_node'] == 'agent' and (text := step.text()):
                print(text, end='')
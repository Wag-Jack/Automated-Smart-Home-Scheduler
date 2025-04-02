import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Retrieve OpenAI API key from .env file
load_dotenv()
key = os.getenv('OPENAI_API_KEY')

#Define tools later for use with Home Assistant / SQLite
tools = []

# Model configuration
model = init_chat_model("gpt-4o-mini", model_provider="openai")

# Create agent from configured model
scheduler_agent = create_react_agent(model, tools)

# Send initial intro message using scheduler agent, showing each token in execution
i = open('introduction.txt', 'r')
intro = i.read()

for step, metadata in scheduler_agent.stream(
    {"messages": [HumanMessage(content=intro)]},
    stream_mode="messages"
):
    if metadata['langgraph_node'] == 'agent' and (text := step.text()):
        print(text, end='|')

#TODO: SQLite interaction

# Below creates an agent that remembers its state, will come back to this in integration
"""
def send_query(agent, config):
    query = input('What would you like the scheduler agent to do? | ')
   
    for chunk in agent.stream(
        {"messages": [HumanMessage(content=query)]}, config
    ):
        print(chunk)
        print('---')
"""
        
"""
memory = MemorySaver()
scheduler_agent = create_react_agent(model, tools, checkpointer=memory)"

config = {"configurable": {"thread_id": "cis489"}}
"""
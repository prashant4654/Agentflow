# quick_test.py
import os
from dotenv import load_dotenv
from agentflow.graph import StateGraph, Agent
# from agentflow.utils import END
from agentflow.state import AgentState, Message

# Load API key
load_dotenv()

# Create agent
agent = Agent(
    model="google/gemini-2.5-flash-lite",
    system_prompt=[{"role": "system", "content": "You are a helpful assistant"}]
)

# Build workflow
graph = StateGraph()
graph.add_node("agent", agent)
graph.set_entry_point("agent")
# graph.add_edge("agent", END)

# Compile and run
app = graph.compile()
result = app.invoke({
    "messages": [Message.text_message("Say hello!", "user")],
})

print("Response:", result["messages"][-1].content)
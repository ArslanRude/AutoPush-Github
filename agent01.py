from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from pydantic import BaseModel,Field
from create_workflow import workflow_chain
from generate_code import generate_code_chain

model = ChatGroq(model_name="llama-3.3-70b-versatile",temperature=0.6,max_retries=2)

class AgentState(BaseModel):
    topic_history: list[dict[str, str]] = Field(description="The list of previously covered topics") 
    topic: str = Field(description="The topic to generate code for")
    description: str = Field(description="The description of the topic")
    workflow: str = Field(description="The workflow to generate code alonge with file name for the topic")
    code: dict[str, str] = Field(description="The code generated for the topic")

graph = StateGraph(AgentState)

def create_workflow(state:AgentState):
    response = workflow_chain.invoke({"topic": state.topic, "description": state.description})
    return {"workflow": response}

def generate_code(state:AgentState):
    response = generate_code_chain.invoke({"topic": state.topic, "description": state.description, "workflow": state.workflow})
    return {"code": response}

graph.add_node("create_workflow", create_workflow)
graph.add_node("generate_code", generate_code)

graph.set_entry_point("create_workflow")
graph.add_edge("create_workflow", "generate_code")
graph.add_edge("generate_code", END)

app = graph.compile()




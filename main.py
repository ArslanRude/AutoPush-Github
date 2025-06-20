from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from pydantic import Field
from langgraph.graph import StateGraph, END
from create_topic_description import create_chain
from critique_topic_description import critique_chain

class ReflectionState(BaseModel):
    topic_history: list[dict[str, str]] = Field(description="The list of previously covered topics") 
    topic: str = Field(description="The topic to generate code for")
    description: str = Field(description="The description of the topic")
    critique: str = Field(description="The critique of the topic")  
    count: int = Field(description="The count of topics generated")

graph = StateGraph(ReflectionState)

def create_topic_description(state:ReflectionState):
    response = create_chain.invoke({"topic_history": state.topic_history, "topic": state.topic, "description": state.description, "critique": state.critique})
    lines = response.strip().split('\n')
    topic = lines[0].replace('topic:', '').strip()
    description = '\n'.join(lines[2:]).replace('description:', '').strip()
    return {"topic": topic, "description": description, "count": state.count + 1}

def critique_topic_description(state:ReflectionState):
    critique = critique_chain.invoke({"topic_history": state.topic_history, "topic": state.topic, "description": state.description})
    return {"critique": critique}

def should_continue(state:ReflectionState):
    if state.count < 2:
        return "critique_topic_description"
    return END

graph.add_node("create_topic_description", create_topic_description)
graph.add_node("critique_topic_description", critique_topic_description)

graph.set_entry_point("create_topic_description")
graph.add_conditional_edges("create_topic_description", should_continue)
graph.add_edge('critique_topic_description', 'create_topic_description')

app = graph.compile()

initial_state = {
    "topic_history": [],
    "topic": "",
    "description": "",
    "critique": "",
    "count": 0
}

result = app.invoke(initial_state)

print("\nFinal State:")
print("-" * 50)
print(f"Topic: {result['topic']}")
print(f"Description: {result['description']}")
print(f"Critique: {result['critique']}")
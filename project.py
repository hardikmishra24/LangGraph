import os
from typing import TypedDict

# Creating State.
# A state is a dictionary that store the infomation from start to the end of the graph.
# A state is a informaiton which flows through the nodes.

class pipelinestate(TypedDict):
    raw_input : str
    edited_text : str
    script_text : str
    final_output : str


# A node is a python function that perfoms a specific task using the information avaialable in the state.
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)

def editor_node(state: pipelinestate)-> dict:
    """Stage 1: Cleans up grammar, removes typoss, and refines the tone."""

    prompt = (
        "You are an expert copyeditor. Clean up the following raw text."
        "Fix any grammatical errors, spelling mistakes, and smooth out the transition flow."
        "While keeping the core message intact. Return only the edited text.\n\n"
        f"Text:\n{state['raw_input']}"
    )
    response = llm.invoke(prompt)

    return {"edited_text": response.content.strip()}

def scriptwriter_node(state: pipelinestate)-> dict:
    """Stage 2: Converts the edited text into a script format."""

    prompt = (
        "You are a professional scriptwriter. Convert the following edited text into a script format."
        "Ensure that the dialogue is clear, and the scenes are well-structured."
        "Return only the script text.\n\n"
        f"Edited Text:\n{state['edited_text']}"
    )
    response = llm.invoke(prompt)

    return {"script_text": response.content.strip()}

def translator_node(state: pipelinestate)-> dict:
    """Stage 3: Translates the script into Spanish."""

    prompt = (
        "You are a skilled translator. Translate the following script into Hindi."
        "Ensure that the translation maintains the original meaning and tone."
        "Return only the translated script.\n\n"
        f"Script Text:\n{state['script_text']}"
    )
    response = llm.invoke(prompt)

    return {"final_output": response.content.strip()}
# An edge is a connection between the two nodes that determines which node should be executed next. 

from langgraph.graph import StateGraph, START, END

# Creating the graph.
graph = StateGraph(pipelinestate)
graph.add_node("editor", editor_node)
graph.add_node("scriptwriter", scriptwriter_node)
graph.add_node("translator", translator_node)

#Add the edges
graph.add_edge(START, "editor")
graph.add_edge("editor", "scriptwriter")
graph.add_edge("scriptwriter", "translator")
graph.add_edge("translator", END)

app = graph.compile()

result = app.invoke({"raw_input": "This is a sample text that needs to be edited, converted into a script, and then translated into hindi."})

print(result["final_output"])
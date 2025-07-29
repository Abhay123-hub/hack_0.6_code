from langgraph.graph import StateGraph,START,END
from typing import List
from typing_extensions import TypedDict


class State(TypedDict):
    questions:List[str]
    documents:str
    relevant_documents:List[List[str]]
    responses:List[str]
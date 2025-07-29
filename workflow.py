from langgraph.graph import END, StateGraph, START
from Agent import Agent
from State import State

class workflow:
    def __init__(self):
        self.agent = Agent()
    def create_workflow(self):
        workflow = StateGraph(State)
        workflow.add_node("get_relevant_documents",self.agent.get_relevant_documents)
        workflow.add_node("get_response",self.agent.get_response)

        workflow.add_edge(START,"get_relevant_documents")
        workflow.add_edge("get_relevant_documents","get_response")
        workflow.add_edge("get_response",END)
        app = workflow.compile()
        return app
    
    def execute(self,input_dict): ## i really need to know that what should it take as input and what should it give as an output
        app = self.create_workflow()
        response = app.invoke(input_dict)
        return response
    

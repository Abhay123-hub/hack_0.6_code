from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from workflow import workflow  # Import your LangGraph workflow

app = FastAPI()

# Request body model
class InputData(BaseModel):
    documents: str
    questions: List[str]



@app.post("/api/v1/hackrx/run")
async def run_hackrx(input_data: InputData):
    try:
        graph = workflow()

        input_dict = {
            "documents": input_data.documents,
            "questions": input_data.questions
        }
       
        result = graph.execute(input_dict)

        if "responses" in result:
            return {"answers": result["responses"]}
        else:
            raise ValueError("Invalid workflow response structure.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


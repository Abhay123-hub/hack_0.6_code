from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from workflow import workflow

# Initialize FastAPI
app = FastAPI()

# Authorization token
FIXED_TOKEN = "8c8772aa86766c163300f81f3a2dbc6fea8541aa11f50bf95c5540d1987687bd"
security = HTTPBearer()

# Validate Authorization header
def validate_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != FIXED_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# Request body model
class InputData(BaseModel):
    documents: str
    questions: List[str]

@app.post("/api/v1/hackrx/run")
async def run_hackrx(input_data: InputData, _: str = Depends(validate_token)):
    try:
        # Prepare input dictionary for the workflow
        input_dict = {
            "documents": input_data.documents,
            "questions": input_data.questions
        }

        # Run your AI workflow (agent handles downloading internally)
        graph = workflow()
        result = graph.execute(input_dict)

        # Validate workflow response
        if "responses" in result:
            return {"answers": result["responses"]}
        else:
            raise HTTPException(status_code=500, detail="Invalid workflow response structure.")

    except HTTPException as he:
        # Pass HTTPExceptions directly
        raise he
    except Exception as e:
        # Catch all other errors
        raise HTTPException(status_code=500, detail=str(e))





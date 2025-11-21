# agent_service.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from run_agent import run_agent_query

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/agent")
async def agent_endpoint(req: QueryRequest):
    return run_agent_query(req.query)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
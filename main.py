from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from engine import CivicClassifier
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
ai_engine = CivicClassifier()

class ReportRequest(BaseModel):
    description: str

@app.post("/analyze-issue")
async def analyze_issue(request: ReportRequest):
    try:
        analysis = ai_engine.tag_issue(request.description)
        
        # # Priority Logic: Still based on the predicted tag
        # priority = "Medium"
        # if analysis["confidence"] > 0.8:
        #     if analysis["tag"] in ["public_safety", "water_leak", "power_outage"]:
        #         priority = "High"
        
        return {
            "tag": analysis["tag"],
            "department": analysis["department"],
            "confidence": analysis["confidence"]
        }
    except Exception as e:
        print(f"DEBUG ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "civic-saathi-ai"}

@app.get("/")
async def root():
    return {"message": "Civic Saathi AI Service is live. Send POST requests to /analyze-issue"}

@app.get("/health")
def health_check():
    return {"status": "AI Service is Online"}

from fastapi import FastAPI
from pydantic import BaseModel

from src.recommend import recommend_for_user
from src.multimodal import search_images_by_text

app = FastAPI(title="Healthcare Multimodal AI Agent (Qdrant)")


class AskRequest(BaseModel):
    user_id: str
    question: str


@app.get("/")
def root():
    return {"message": "Healthcare AI Agent is running ✅"}


@app.post("/ask")
def ask(req: AskRequest):
    # Text search + memory + recommendations
    result = recommend_for_user(req.user_id, req.question)

    # Multimodal: image retrieval using the same query text
    result["image_matches"] = search_images_by_text(req.question, limit=3)

    return result

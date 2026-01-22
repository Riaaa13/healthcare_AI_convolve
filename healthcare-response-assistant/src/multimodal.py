from qdrant_client import QdrantClient
import torch
from transformers import CLIPProcessor, CLIPModel

client = QdrantClient(url="http://localhost:6333")
COLLECTION = "medical_images"

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def search_images_by_text(query: str, limit: int = 3):
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)

    qvec = text_features[0].cpu().numpy().tolist()

    results = client.query_points(
        collection_name=COLLECTION,
        query=qvec,
        limit=limit,
        with_payload=True
    )

    image_hits = []
    for p in results.points:
        image_hits.append({
            "file_name": p.payload.get("file_name"),
            "path": p.payload.get("path"),
            "score": p.score
        })

    return image_hits

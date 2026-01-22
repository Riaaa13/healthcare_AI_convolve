from qdrant_client import QdrantClient
import torch
from transformers import CLIPProcessor, CLIPModel

client = QdrantClient(url="http://localhost:6333")

COLLECTION = "medical_images"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

query = input("Enter text to search images: ")

inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)

with torch.no_grad():
    text_features = model.get_text_features(**inputs)

qvec = text_features[0].cpu().numpy().tolist()

results = client.query_points(
    collection_name=COLLECTION,
    query=qvec,
    limit=3,
    with_payload=True
)

print("\n🖼️ Top matching images:")
for p in results.points:
    print("-", p.payload.get("file_name"), "| score:", p.score)

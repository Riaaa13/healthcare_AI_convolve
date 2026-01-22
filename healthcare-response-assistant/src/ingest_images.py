import os
from pathlib import Path
from PIL import Image

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

import torch
from transformers import CLIPProcessor, CLIPModel

client = QdrantClient(url="http://localhost:6333")

COLLECTION = "medical_images"
IMG_DIR = Path("data/images")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

points = []
pid = 1

for img_file in os.listdir(IMG_DIR):
    path = IMG_DIR / img_file
    if path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
        continue

    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        img_features = model.get_image_features(**inputs)

    img_vec = img_features[0].cpu().numpy().tolist()

    points.append(
        PointStruct(
            id=pid,
            vector=img_vec,
            payload={
                "file_name": img_file,
                "path": str(path),
                "type": "medical_image"
            }
        )
    )
    pid += 1

if points:
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"✅ Ingested {len(points)} images into {COLLECTION}")
else:
    print("⚠️ No images found to ingest.")

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import time

client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION = "patient_memory"

def store_memory(user_id: str, memory_text: str):
    vec = model.encode(memory_text).tolist()
    point = PointStruct(
        id=int(time.time()),  # simple unique id
        vector=vec,
        payload={
            "user_id": user_id,
            "memory_text": memory_text,
            "timestamp": time.time()
        }
    )
    client.upsert(collection_name=COLLECTION, points=[point])
    print("✅ Memory stored!")

def get_memory(user_id: str):
    results = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        ),
        limit=5
    )
    return results[0]

# Demo
uid = input("Enter user id: ")
choice = input("1=store memory, 2=get memory: ")

if choice == "1":
    text = input("Enter memory (allergy/condition/etc): ")
    store_memory(uid, text)
else:
    memories = get_memory(uid)
    print("\n🧠 Memories:")
    for m in memories:
        print("-", m.payload["memory_text"])

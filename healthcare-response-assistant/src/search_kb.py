from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION = "medical_knowledge_base"

query = input("Ask a health question: ")

qvec = model.encode(query).tolist()

results = client.query_points(
    collection_name=COLLECTION,
    query=qvec,
    limit=3,
    with_payload=True
)

print("\n🔍 Top Results:")
for point in results.points:
    print("\n---")
    print("Score:", point.score)
    print("Source:", point.payload.get("source_name"))
    print("Text:", point.payload.get("text")[:300], "...")

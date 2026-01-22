from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
print("✅ Connected!")
print(client.get_collections())

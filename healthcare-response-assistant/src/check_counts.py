from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

for col in ["medical_knowledge_base", "patient_memory", "health_resources"]:
    info = client.get_collection(col)
    print(col, "=> points:", info.points_count)

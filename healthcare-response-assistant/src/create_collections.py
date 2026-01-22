from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(url="http://localhost:6333")

EMBEDDING_DIM = 384  # sentence-transformers dimension

client.recreate_collection(
    collection_name="medical_knowledge_base",
    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
)

client.recreate_collection(
    collection_name="health_resources",
    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
)

client.recreate_collection(
    collection_name="patient_memory",
    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
)
client.recreate_collection(
    collection_name="medical_images",
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)


print("✅ Collections created successfully!")

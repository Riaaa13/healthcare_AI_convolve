from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer


client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer("all-MiniLM-L6-v2")

KB_COLLECTION = "medical_knowledge_base"
MEM_COLLECTION = "patient_memory"
RES_COLLECTION = "health_resources"


def get_patient_memory(user_id: str, limit: int = 5):
    """Fetch last few memory entries for user"""
    res = client.scroll(
        collection_name=MEM_COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        ),
        limit=limit
    )
    return [p.payload.get("memory_text", "") for p in res[0]]


def search_kb(query: str, limit: int = 3):
    """Search medical knowledge base"""
    qvec = model.encode(query).tolist()
    results = client.query_points(
        collection_name=KB_COLLECTION,
        query=qvec,
        limit=limit,
        with_payload=True
    )
    return results.points


def basic_safety_checks(query: str):
    """Detect emergency keywords (simple triage)"""
    emergency_words = [
        "chest pain", "breathing", "unconscious", "seizure",
        "heavy bleeding", "stroke", "heart attack", "suicide"
    ]
    for w in emergency_words:
        if w in query.lower():
            return True
    return False


def generate_recommendations(user_query: str, memory_list, kb_results):
    """
    Recommendations grounded from retrieved guideline chunks.
    Memory is used dynamically (no hardcoded disease rules).
    """
    recs = []

    # Emergency routing
    if basic_safety_checks(user_query):
        recs.append("⚠️ This may be an emergency. Please call 112 / 108 or visit the nearest hospital immediately.")

    # ✅ Memory usage (dynamic)
    if memory_list:
        recs.append("🧠 Patient context from memory (used to guide response):")
        for m in memory_list[-5:]:
            recs.append(f"• {m}")

    # ✅ Guideline-based recommendations (from Qdrant search)
    if kb_results:
        recs.append("📌 Key guideline-based points:")
        for r in kb_results[:3]:
            txt = r.payload.get("text", "").strip()
            if txt:
                recs.append("• " + txt[:180] + ("..." if len(txt) > 180 else ""))
    else:
        recs.append("⚠️ I couldn't find matching guideline text in the knowledge base for this query.")

    # General safe advice
    recs.append("✅ If symptoms worsen or persist for more than 2–3 days, consult a licensed doctor.")

    # Evidence sources
    evidence = []
    for r in kb_results:
        evidence.append({
            "source": r.payload.get("source_name"),
            "page_no": r.payload.get("page_no"),
            "score": r.score,
        })

    return recs, evidence


def recommend_for_user(user_id: str, user_query: str):
    memory_list = get_patient_memory(user_id=user_id)
    kb_results = search_kb(user_query)

    recs, evidence = generate_recommendations(user_query, memory_list, kb_results)

    return {
        "user_id": user_id,
        "query": user_query,
        "patient_memory_used": memory_list,
        "recommendations": recs,
        "evidence_sources": evidence
    }


if __name__ == "__main__":
    uid = input("Enter user id: ")
    q = input("Ask a health question: ")

    output = recommend_for_user(uid, q)

    print("\n🟢 Recommendations:")
    for i, r in enumerate(output["recommendations"], 1):
        print(f"{i}. {r}")

    print("\n📌 Evidence Sources Used:")
    for e in output["evidence_sources"]:
        print(f"- {e['source']} (page {e['page_no']}), score={e['score']:.3f}")

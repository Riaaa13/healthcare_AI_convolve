import os
from pathlib import Path
from pypdf import PdfReader

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer("all-MiniLM-L6-v2")

DATA_DIR = Path("data/text")
COLLECTION = "medical_knowledge_base"


def chunk_text(text, chunk_size=800, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks


def read_pdf_pages(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.replace("\n", " ").strip()
        if text:
            pages.append((i + 1, text))
    return pages


def read_txt_file(txt_path: Path):
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\n", " ").strip()
    return text


points = []
point_id = 1

for file in os.listdir(DATA_DIR):
    file_path = DATA_DIR / file
    ext = file_path.suffix.lower()

    # ✅ PDF ingestion
    if ext == ".pdf":
        print(f"📄 Reading PDF: {file}")
        pages = read_pdf_pages(file_path)

        for page_no, page_text in pages:
            chunks = chunk_text(page_text)

            for chunk_id, chunk in enumerate(chunks):
                vector = model.encode(chunk).tolist()

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "source_type": "pdf",
                            "source_name": file,
                            "page_no": page_no,
                            "chunk_id": chunk_id,
                            "text": chunk
                        }
                    )
                )
                point_id += 1

    # ✅ TXT ingestion (Hugging Face dataset files)
    elif ext == ".txt":
        print(f"📝 Reading TXT: {file}")
        text = read_txt_file(file_path)
        chunks = chunk_text(text)

        for chunk_id, chunk in enumerate(chunks):
            vector = model.encode(chunk).tolist()

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "source_type": "txt",
                        "source_name": file,
                        "page_no": None,
                        "chunk_id": chunk_id,
                        "text": chunk
                    }
                )
            )
            point_id += 1

    else:
        # Skip other file types
        continue


if points:
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"\n✅ Successfully ingested {len(points)} chunks into {COLLECTION}")
else:
    print("\n⚠️ No PDF/TXT chunks found to ingest.")

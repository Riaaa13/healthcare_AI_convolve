# Healthcare AI Agent (Qdrant + FastAPI)

A simple Healthcare Response Assistant that supports:
- Medical guideline/document search (PDF + TXT)
- Patient memory storage
- Query answering via a FastAPI endpoint



---

## Project Structure
healthcare-response-assistant/
├── data/
│ ├── text/ # PDFs + .txt files (guidelines, medical info)
│ └── images/ # optional (for multimodal)
├── src/
│ ├── app.py
│ ├── create_collections.py
│ ├── ingest_kb.py
│ ├── search_kb.py
│ ├── memory.py
│ ├── recommend.py
│ ├── ingest_images.py # optional
│ ├── multimodal.py # optional
│ └── download_hf_images.py # optional


---

## Requirements

- Python 3.10+
- Docker Desktop

---

## Setup Instructions

### 1) Clone and open the project. Run the below code in bash terminal of VS CODE 
git clone <your-repo-link>
cd healthcare-response-assistant

### 2)Create and activate virtual environment
for (Windows (CMD))
-python -m venv venv
-venv\Scripts\activate.bat
### 3)Install dependencies 
-pip install qdrant-client sentence-transformers fastapi uvicorn pypdf pillow torch torchvision transformers datasets


### 4) Run Qdrant (Docker)
4) Start Qdrant
-Open a new terminal and run the following:
-docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant

### 5) Create Collections
 Create Qdrant collections and then check by running this in your terminal:
-python src/create_collections.py

### 6)Add Documents (PDF/TXT)
 Add your files to data/text/

Put your guideline PDFs and/or .txt files here:
-data/text/
-Examples include :
  example_guideline.pdf
  example_notes.txt

### 7)Ingest Documents into Qdrant
 Ingest the knowledge base and then run this in your terminal
-python src/ingest_kb.py

-Run this again whenever you add/update files in data/text/.

### 8)Test Search (Optional)
Test knowledge base search by running this in terminal:
-python src/search_kb.py

-Example query:

fever precautions
mental health support

### 9)Store Patient Memory (Optional)
 Store memory for a user
-python src/memory.py


Example memory:

"I have diabetes"
"I am allergic to ibuprofen"

### 10)Run the API (FastAPI)
 Start FastAPI server
-uvicorn src.app:app --reload


-Open Swagger UI:
-http://127.0.0.1:8000/docs

Ask a Query (API)
11) Use /ask endpoint

In /docs → POST /ask:

{
  "user_id": "Rhythm",
  "question": "fever precautions"
}
And you will get your response in the response box in the link opened :)


from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path("data/text")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# You can change this dataset if needed
DATASET_NAME = "pubmed_qa"
SPLIT = "train"

def main(limit=200):
    print(f"📥 Downloading {DATASET_NAME} ({SPLIT}) from Hugging Face...")
    ds = load_dataset(DATASET_NAME, "pqa_labeled", split=SPLIT)

    print("✅ Loaded dataset!")
    print("Columns:", ds.column_names)

    saved = 0
    for i in range(min(limit, len(ds))):
        row = ds[i]

        question = row.get("question", "")
        context = row.get("context", "")
        long_answer = row.get("long_answer", "")

        text = f"""
SOURCE: {DATASET_NAME}
ID: {i}

QUESTION:
{question}

CONTEXT:
{context}

LONG_ANSWER:
{long_answer}
""".strip()

        file_path = OUT_DIR / f"pubmedqa_{i}.txt"
        file_path.write_text(text, encoding="utf-8")

        saved += 1

    print(f"🎉 Saved {saved} medical text files into: {OUT_DIR}")

if __name__ == "__main__":
    main(limit=200)

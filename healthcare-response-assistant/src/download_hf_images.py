from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path("data/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main(limit=30):
    print("📥 Loading dataset...")
    ds = load_dataset("hf-vision/chest-xray-pneumonia")

    train = ds["train"]

    saved = 0
    for i in range(min(limit, len(train))):
        img = train[i]["image"]
        label = train[i].get("label", "unknown")

        path = OUT_DIR / f"xray_{i}_label_{label}.jpg"
        img.convert("RGB").save(path)

        saved += 1

    print(f"✅ Saved {saved} images into {OUT_DIR}")

if __name__ == "__main__":
    main(limit=30)

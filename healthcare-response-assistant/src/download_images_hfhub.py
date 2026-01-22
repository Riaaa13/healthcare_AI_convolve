import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

OUT_DIR = Path("data/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# You can change this repo if needed
REPO_ID = "hf-vision/chest-xray-pneumonia"

def main(limit=50):
    print("📥 Downloading repository from Hugging Face Hub...")
    local_repo_path = snapshot_download(repo_id=REPO_ID)

    print("✅ Repo downloaded at:", local_repo_path)

    copied = 0
    for root, _, files in os.walk(local_repo_path):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                src = Path(root) / f
                dst = OUT_DIR / f"{copied}_{f}"
                shutil.copy(src, dst)
                copied += 1
                if copied >= limit:
                    break
        if copied >= limit:
            break

    print(f"🎉 Copied {copied} images into: {OUT_DIR}")

if __name__ == "__main__":
    main(limit=50)


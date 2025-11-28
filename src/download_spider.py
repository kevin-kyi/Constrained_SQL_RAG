import io
import os
import shutil
import zipfile

import requests


SPIDER_ZIP_URL = "https://github.com/taoyds/spider/archive/refs/heads/master.zip"
TARGET_ROOT = "spider_dataset"
TARGET_SUBDIR = "spider"  # Data dir name


def main():
    os.makedirs(TARGET_ROOT, exist_ok=True)
    target_dir = os.path.join(TARGET_ROOT, TARGET_SUBDIR)

    if os.path.isdir(target_dir):
        print(f"[INFO] Spider dataset already present at: {target_dir}")
        return

    print(f"[INFO] Downloading Spider 1.0 from {SPIDER_ZIP_URL} ...")
    resp = requests.get(SPIDER_ZIP_URL)
    resp.raise_for_status()

    print("[INFO] Extracting zip into spider_dataset/ ...")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(TARGET_ROOT)

    # GitHub archive extracts to spider-master/ by default
    extracted_root = os.path.join(TARGET_ROOT, "spider-master")
    if not os.path.isdir(extracted_root):
        raise RuntimeError(f"Expected extracted dir {extracted_root} not found.")

    # Rename spider-master -> spider for a nice stable path
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.rename(extracted_root, target_dir)

    print(f"[INFO] Done. Spider dataset available at: {target_dir}")
    print("[INFO] Key files:")
    print(f"  - {os.path.join(target_dir, 'tables.json')}")
    print(f"  - {os.path.join(target_dir, 'train_spider.json')}")
    print(f"  - {os.path.join(target_dir, 'dev.json')}")
    print(f"  - {os.path.join(target_dir, 'database')}/<db_id>/*.sqlite")


if __name__ == "__main__":
    main()

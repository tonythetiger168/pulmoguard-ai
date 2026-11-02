"""
luna25_extract_blocks.py — 解壓 LUNA25 nodule blocks 並整理成 nodule_blocks/{image,metadata}/*.npy

用法：python scripts/luna25_extract_blocks.py <zip> <dest_dir>

壓縮檔內的資料夾名稱與層級沒有文件說明（第一次 RunPod 試跑就因為找不到 image/ 而中止），
所以這裡先把壓縮檔結構印到 log，再依內容（.npy 數量、資料夾名稱）找出影像與 metadata 資料夾。
"""
from __future__ import annotations

import collections
import os
import shutil
import sys
import zipfile


def summarize(names: list[str]) -> None:
    dirs = collections.Counter(os.path.dirname(n) for n in names if not n.endswith("/"))
    exts = collections.Counter(os.path.splitext(n)[1].lower() for n in names if not n.endswith("/"))
    print(f"zip: {len(names)} entries; extensions {dict(exts.most_common(6))}")
    for d, c in dirs.most_common(8):
        print(f"  {c:6d} files in {d or '.'}/")
    print("  first entries:", names[:5])


def extract_nested(dest: str) -> None:
    """壓縮檔裡若還有 zip（例如 image.zip），再解一層"""
    for root, _, files in os.walk(dest):
        for f in files:
            if f.endswith(".zip"):
                p = os.path.join(root, f)
                print(f"nested zip: {p}")
                with zipfile.ZipFile(p) as z:
                    z.extractall(root)
                os.remove(p)


def find_dirs(dest: str) -> tuple[str | None, str | None]:
    counts = {}
    for root, _, files in os.walk(dest):
        n = sum(f.endswith(".npy") for f in files)
        if n:
            counts[root] = n
    for d, n in sorted(counts.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  {n:6d} .npy in {d}")
    def best(key):
        cands = [d for d in counts if key in os.path.basename(d).lower()]
        return max(cands, key=counts.get) if cands else None
    return best("image"), best("meta")


def main(zip_path: str, dest: str) -> int:
    with zipfile.ZipFile(zip_path) as z:
        summarize(z.namelist())
        tmp = os.path.join(dest, "_extract")
        z.extractall(tmp)
    extract_nested(tmp)
    img, meta = find_dirs(tmp)
    print(f"image dir: {img}\nmetadata dir: {meta}")
    if not img or not meta:
        print("ERROR: could not find image/ and metadata/ folders of .npy files (see listing above)")
        return 1
    out = os.path.join(dest, "nodule_blocks")
    os.makedirs(out, exist_ok=True)
    for src, name in ((img, "image"), (meta, "metadata")):
        target = os.path.join(out, name)
        if os.path.exists(target):
            shutil.rmtree(target)
        shutil.move(src, target)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"nodule_blocks: {len(os.listdir(os.path.join(out, 'image')))} image, "
          f"{len(os.listdir(os.path.join(out, 'metadata')))} metadata")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))

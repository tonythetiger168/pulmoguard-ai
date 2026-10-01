"""
download_tcia.py — 在有外網的伺服器上執行：下載 LIDC-IDRI / LUNA16 訓練資料
用法：
  python download_tcia.py --collection LIDC-IDRI --limit 50        # 先下載 50 例 pilot
  python download_tcia.py --collection LIDC-IDRI --out data/lidc   # 全量 1,018 例（約 124 GB）
  python download_tcia.py --collection LIDC-IDRI --manifest only   # 只產 manifest，交 NBIA Data Retriever

TCIA 免帳號、無審查；NLST 臨床資料仍須另行申請 CDAS。
"""
import argparse
import io
import os
import sys
import time
import zipfile

import requests

NBIA = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


def get_series(collection: str, limit: int | None) -> list[str]:
    """取得 collection 內的 SeriesInstanceUID 清單"""
    r = requests.get(f"{NBIA}/getSeries", params={"Collection": collection},
                     timeout=120)
    r.raise_for_status()
    uids = [s["SeriesInstanceUID"] for s in r.json()]
    return uids[:limit] if limit else uids


def download_series(uid: str, out_dir: str, retries: int = 3) -> str:
    """下載單一 series（zip），已存在則跳過 → 可中斷續傳"""
    path = os.path.join(out_dir, f"{uid}.zip")
    if os.path.exists(path):
        return path
    for i in range(retries):
        try:
            r = requests.get(f"{NBIA}/getImage", params={"SeriesInstanceUID": uid},
                             stream=True, timeout=600)
            r.raise_for_status()
            with open(path + ".tmp", "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            os.replace(path + ".tmp", path)
            return path
        except requests.RequestException as e:
            print(f"  retry {i+1}/{retries} {uid}: {e}", file=sys.stderr)
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"fail: {uid}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--collection", default="LIDC-IDRI")
    p.add_argument("--out", default="data/raw")
    p.add_argument("--limit", type=int, default=50,
                   help="pilot 批次數；設 0 為全量")
    p.add_argument("--manifest", choices=["no", "only"], default="no",
                   help="only=只輸出 manifest.txt 供 NBIA Data Retriever 使用")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    uids = get_series(args.collection, args.limit or None)
    print(f"{args.collection}: {len(uids)} series")

    if args.manifest == "only":
        with open("manifest.txt", "w") as f:
            f.write("Series UID\n" + "\n".join(uids) + "\n")
        print("manifest.txt 已產出，請以 NBIA Data Retriever 開啟下載")
        return

    for i, uid in enumerate(uids):
        download_series(uid, args.out)
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(uids)}] done")
    print("ALL DOWNLOADED → 下一步: python preprocess.py")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# download_luna25.sh — 從 Zenodo 下載 LUNA25 官方 nodule blocks（約 6.8 GB）與標註 CSV
# 用法：bash scripts/download_luna25.sh data/luna25
# 影像 https://zenodo.org/records/14223624（CC BY 4.0）
# 標註 https://zenodo.org/records/14673658（CC BY-NC 4.0：僅限研究，不可用於商用產品訓練）
# 完整 221 GB 影像（luna25_images.zip.001–046）這裡刻意不下載。
set -euo pipefail
DEST=${1:-data/luna25}
IMG=https://zenodo.org/records/14223624/files
ANN=https://zenodo.org/records/14673658/files
CSV=LUNA25_Public_Training_Development_Data.csv
CSV_MD5=${CSV_MD5:-1349fcd9fdd120a990f43a2301f1609b}
mkdir -p "$DEST"; cd "$DEST"

get() {  # 可續傳、失敗重試；完成才改名，避免半個檔被當成完整
  if [ -f "$2" ]; then echo "skip $2"; return 0; fi
  for i in 1 2 3 4 5; do
    wget -c -q --show-progress --tries=3 --timeout=60 -O "$2.part" "$1/$2?download=1" && break
    echo "retry $i for $2"; sleep $((i * 10))
  done
  [ -s "$2.part" ] || { echo "ERROR: download failed: $2" >&2; exit 1; }
  mv "$2.part" "$2"
}

get "$ANN" "$CSV"
echo "$CSV_MD5  $CSV" | md5sum -c - || { echo "ERROR: CSV md5 mismatch" >&2; exit 1; }

if [ -d nodule_blocks/image ] && [ "$(ls nodule_blocks/image | wc -l)" -gt 6000 ]; then
  echo "skip nodule_blocks (already extracted)"
else
  get "$IMG" luna25_nodule_blocks.zip.001
  get "$IMG" luna25_nodule_blocks.zip.002
  # 分割檔是直接切開的單一 zip：先接回再解壓
  cat luna25_nodule_blocks.zip.0?? > luna25_nodule_blocks.zip
  if command -v unzip >/dev/null; then unzip -oq luna25_nodule_blocks.zip
  else python3 -c "import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall('.')" luna25_nodule_blocks.zip
  fi
  # 壓縮檔內的資料夾層級不一定是 nodule_blocks/，找到 image/ 與 metadata/ 的上一層
  if [ ! -d nodule_blocks/image ]; then
    d=$(dirname "$(find . -type d -name image -not -path './nodule_blocks/*' | head -1)")
    [ -n "$d" ] && [ -d "$d/metadata" ] && mv "$d" nodule_blocks
  fi
  n=$(ls nodule_blocks/image 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] || { echo "ERROR: no nodule blocks after extraction" >&2; find . -maxdepth 3 -type d >&2; exit 1; }
  rm -f luna25_nodule_blocks.zip luna25_nodule_blocks.zip.00?   # 解壓成功才刪，省磁碟
fi
echo "LUNA25 ready in $DEST: $(ls nodule_blocks/image | wc -l) blocks, $(($(wc -l < "$CSV") - 1)) annotations"

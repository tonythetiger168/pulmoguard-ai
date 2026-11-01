#!/usr/bin/env bash
# download_luna16.sh — 從 Zenodo 下載 LUNA16（CC BY 4.0）指定 subset 與標註檔
# 用法：bash scripts/download_luna16.sh data/luna16 0          # 最小試跑：只抓 subset0（6.8 GB）
#       bash scripts/download_luna16.sh data/luna16 0 1 2 3   # 多個 subset
# Part 1（subset0–6）：https://zenodo.org/records/3723295
# Part 2（subset7–9）：https://zenodo.org/records/4121926
set -euo pipefail
DEST=${1:-data/luna16}; shift || true
SUBSETS=${*:-0}
P1=https://zenodo.org/records/3723295/files
P2=https://zenodo.org/records/4121926/files
mkdir -p "$DEST"; cd "$DEST"

get() {  # 可續傳下載
  [ -f "$2" ] && [ ! -f "$2.part" ] && { echo "skip $2"; return; }
  wget -c -q --show-progress -O "$2.part" "$1/$2?download=1" && mv "$2.part" "$2"
}

get "$P1" annotations.csv
get "$P1" candidates_V2.zip && unzip -oq candidates_V2.zip && find . -name candidates_V2.csv -not -path ./candidates_V2.csv -exec mv {} . \; || true
get "$P1" evaluationScript.zip && unzip -oq evaluationScript.zip && \
  find . -name annotations_excluded.csv -not -path ./annotations_excluded.csv -exec cp {} . \; || true

for s in $SUBSETS; do
  base=$P1; [ "$s" -ge 7 ] && base=$P2
  if [ -d "subset$s" ] && ls "subset$s"/*.mhd >/dev/null 2>&1; then echo "skip subset$s"; continue; fi
  get "$base" "subset$s.zip"
  unzip -oq "subset$s.zip" && rm "subset$s.zip"    # 解壓後刪 zip，省網路磁碟費用
done
ls -1 | head; echo "LUNA16 ready in $DEST"

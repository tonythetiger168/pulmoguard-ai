#!/usr/bin/env bash
# download_luna16.sh — 從 Zenodo 下載 LUNA16（CC BY 4.0）指定 subset 與標註檔
# 用法：bash scripts/download_luna16.sh data/luna16 0          # 最小試跑：只抓 subset0（6.8 GB）
#       bash scripts/download_luna16.sh data/luna16 0 1 2 3   # 多個 subset
# Part 1（subset0–6）：https://zenodo.org/records/3723295
# Part 2（subset7–9）：https://zenodo.org/records/4121926
# 任何下載或解壓失敗都會立即中止（不會「看起來成功」卻沒有 .mhd）。
set -euo pipefail
DEST=${1:-data/luna16}; shift || true
SUBSETS=${*:-0}
P1=https://zenodo.org/records/3723295/files
P2=https://zenodo.org/records/4121926/files
mkdir -p "$DEST"; cd "$DEST"

get() {  # 可續傳下載；失敗即中止
  if [ -f "$2" ] && [ ! -f "$2.part" ]; then echo "skip $2"; return 0; fi
  wget -c -q --show-progress -O "$2.part" "$1/$2?download=1"
  mv "$2.part" "$2"
}

extract() {  # 有 unzip 用 unzip，沒有就用 Python zipfile（部分映像沒裝 unzip）
  if command -v unzip >/dev/null; then
    unzip -oq "$1"
  else
    python3 -c "import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall('.')" "$1"
  fi
}

get "$P1" annotations.csv

get "$P1" candidates_V2.zip
extract candidates_V2.zip
if [ ! -f candidates_V2.csv ]; then
  f=$(find . -name candidates_V2.csv | head -1)
  [ -n "$f" ] && mv "$f" .
fi
[ -f candidates_V2.csv ] || { echo "ERROR: candidates_V2.csv not found after extraction" >&2; exit 1; }

get "$P1" evaluationScript.zip
extract evaluationScript.zip
if [ ! -f annotations_excluded.csv ]; then
  f=$(find . -name annotations_excluded.csv | head -1)
  [ -n "$f" ] && cp "$f" . || echo "warn: annotations_excluded.csv not found (FROC will not ignore excluded findings)"
fi

for s in $SUBSETS; do
  base=$P1; [ "$s" -ge 7 ] && base=$P2
  if ls "subset$s"/*.mhd >/dev/null 2>&1; then echo "skip subset$s"; continue; fi
  get "$base" "subset$s.zip"
  extract "subset$s.zip"
  n=$(ls "subset$s"/*.mhd 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] || { echo "ERROR: no .mhd in subset$s after extraction" >&2; exit 1; }
  echo "subset$s: $n scans"
  rm "subset$s.zip"    # 解壓成功後才刪 zip，省磁碟費用
done
echo "LUNA16 ready in $DEST"

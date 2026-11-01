#!/usr/bin/env bash
# runpod_pilot.sh — 在單一 RunPod pod 上一次跑完最小試跑，跑完自動關機（停止計費）
#   1) 下載 LUNA16 subset0（6.8 GB）  2) 前處理  3) MONAI 預訓練偵測器 FROC/CPM
#   4) 假陽性削減 baseline 訓練（AMP、早停、可續跑）  5) 停止 pod
# 用法（pod 內，repo 根目錄）：
#   bash scripts/runpod_pilot.sh            # 預設資料放 /workspace（pod volume，重啟不遺失）
#   KEEP_POD=1 bash scripts/runpod_pilot.sh # 跑完不關機
set -euo pipefail
WS=${WS:-/workspace}
DATA=$WS/data/luna16
PROC=$WS/data/processed
RUNS=$WS/runs
LOG=$RUNS/pilot_$(date +%Y%m%d_%H%M).log
mkdir -p "$RUNS"
exec > >(tee -a "$LOG") 2>&1
t0=$(date +%s)

stop_pod() {
  echo "elapsed: $(( ($(date +%s) - t0) / 60 )) min. log: $LOG"
  if [ "${KEEP_POD:-0}" != "1" ] && [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null; then
    echo "stopping pod $RUNPOD_POD_ID to stop GPU billing (volume disk is kept)"
    runpodctl stop pod "$RUNPOD_POD_ID"
  fi
}
trap stop_pod EXIT   # 不論成功或失敗都關機，避免閒置計費

pip install -q -r requirements.txt
nvidia-smi --query-gpu=name,memory.total --format=csv || true

echo "== 1/4 download subset0 =="
bash scripts/download_luna16.sh "$DATA" 0

echo "== 2/4 preprocess =="
python -m src.preprocess.pipeline --luna_root "$DATA" --out "$PROC" --subsets 0 --workers "$(nproc)"

echo "== 3/4 pretrained MONAI detector on subset0 (held-out fold for these weights) =="
python -m scripts.eval_monai_detector --luna_root "$DATA" --subset 0 \
  --bundle_dir "$WS/models/bundles" --out "$RUNS/monai_detector" || echo "detector step failed; continuing"

echo "== 4/4 FP-reduction baseline =="
python -m scripts.train_baseline --meta "$PROC/meta.csv" --npy_dir "$PROC/npy" \
  --candidates "$DATA/candidates_V2.csv" --out "$RUNS/baseline" --resume \
  --num_workers "$(( $(nproc) > 8 ? 8 : $(nproc) ))"

echo "== results =="
cat "$RUNS/monai_detector/froc.json" 2>/dev/null || true
python -c "import json;m=json.load(open('$RUNS/baseline/metrics.json'));m.pop('history');print(m)"

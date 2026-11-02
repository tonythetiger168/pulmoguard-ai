#!/usr/bin/env bash
# runpod_fp_chain.sh — 假陽性過濾器接在偵測器後面，看低 FP 區間的敏感度能提升多少
#   1) 下載 LUNA16 subset 0–4  2) 前處理
#   3) 用 subset 1–4 訓練過濾器（subset 4 當驗證；subset 0 完全不碰）
#   4) 取得偵測器在 subset 0 的偵測結果（先用 pilot-results 上已存的，沒有才重跑偵測器）
#   5) 用過濾器重新打分 → 重算 FROC/CPM（scripts/chain_fp_filter.py）
#   6) 結果推到 pilot-results → 刪除 pod
# 用法（pod 內，repo 根目錄）：bash scripts/runpod_fp_chain.sh
#   需要 GH_TOKEN（Contents: write）才會推結果並刪 pod；沒有則只停止 pod
#   DETECTIONS_REF：pilot-results 上偵測結果所在目錄（預設為 CPM 0.888 那一輪）
set -euo pipefail
WS=${WS:-/workspace}
DATA=$WS/data/luna16
PROC=$WS/data/processed
RUNS=$WS/runs
DET_REF=${DETECTIONS_REF:-results/pilot_20261102_0524}
REPO=${RESULTS_REPO:-tonythetiger168/pulmoguard-ai}
LOG=$RUNS/fp_chain_$(date +%Y%m%d_%H%M).log
mkdir -p "$RUNS"
exec > >(tee -a "$LOG") 2>&1
t0=$(date +%s)

save_results() {
  OUT_DIR="results/fp_chain_$(date +%Y%m%d_%H%M)"
  mkdir -p "$OUT_DIR"
  cp "$RUNS/fp_chain/chain_froc.json" "$RUNS/fp_chain/detections_scored.csv" "$OUT_DIR/" 2>/dev/null || true
  cp "$RUNS/fp_filter/metrics.json" "$OUT_DIR/filter_metrics.json" 2>/dev/null || true
  cp "$LOG" "$OUT_DIR/run.log" 2>/dev/null || true
  nvidia-smi --query-gpu=name --format=csv,noheader > "$OUT_DIR/gpu.txt" 2>/dev/null || true
  [ -n "${GH_TOKEN:-}" ] || { echo "GH_TOKEN not set: results kept on the volume only ($OUT_DIR)"; return 1; }
  local auth tmp; auth=$(printf 'x-access-token:%s' "$GH_TOKEN" | base64 -w0); tmp=$(mktemp -d)
  local G=(git -c "http.https://github.com/.extraheader=AUTHORIZATION: basic $auth")
  "${G[@]}" clone -q --depth 1 --branch pilot-results "https://github.com/$REPO.git" "$tmp"
  mkdir -p "$tmp/$OUT_DIR" && cp -r "$OUT_DIR/." "$tmp/$OUT_DIR/"
  git -C "$tmp" add "$OUT_DIR"
  git -C "$tmp" -c user.name="runpod-pilot" -c user.email="runpod-pilot@users.noreply.github.com" \
    commit -qm "fp-chain results $(basename "$OUT_DIR") on $(cat "$OUT_DIR/gpu.txt" 2>/dev/null)"
  # 其他 pod 可能同時推到 pilot-results：被拒就 rebase 後重推
  for i in 1 2 3 4 5; do
    if (cd "$tmp" && "${G[@]}" push -q origin pilot-results); then
      echo "results pushed: https://github.com/$REPO/tree/pilot-results/$OUT_DIR"; return 0
    fi
    sleep $((i * 5))
    (cd "$tmp" && "${G[@]}" pull -q --rebase --depth 50 origin pilot-results) || true
  done
  return 1
}

finish() {
  echo "elapsed: $(( ($(date +%s) - t0) / 60 )) min. log: $LOG"
  local saved=0
  save_results && saved=1
  [ "${KEEP_POD:-0}" = "1" ] && return
  [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null || return
  if [ "$saved" = "1" ]; then
    echo "results are on GitHub: removing pod $RUNPOD_POD_ID"; runpodctl remove pod "$RUNPOD_POD_ID"
  else
    echo "push failed: stopping pod $RUNPOD_POD_ID (volume kept)"; runpodctl stop pod "$RUNPOD_POD_ID"
  fi
}
trap finish EXIT

pip install -q -r requirements.txt
nvidia-smi --query-gpu=name,memory.total --format=csv || true

echo "== 1/5 download subsets 0-4 =="
bash scripts/download_luna16.sh "$DATA" 0 1 2 3 4

echo "== 2/5 preprocess =="
python -m src.preprocess.pipeline --luna_root "$DATA" --out "$PROC" --subsets 0 1 2 3 4

echo "== 3/5 train FP filter on subsets 1-4 (val = subset 4, subset 0 held out) =="
python - "$PROC/meta.csv" "$PROC/meta_no_subset0.csv" <<'PY'
import sys, pandas as pd
m = pd.read_csv(sys.argv[1]); m[m.subset != 0].to_csv(sys.argv[2], index=False)
print("filter training series:", (m.subset != 0).sum(), "| held out (subset 0):", (m.subset == 0).sum())
PY
python -m scripts.train_baseline --meta "$PROC/meta_no_subset0.csv" --npy_dir "$PROC/npy" \
  --candidates "$DATA/candidates_V2.csv" --out "$RUNS/fp_filter" --val_subset 4 --resume --num_workers 4

echo "== 4/5 detector output on subset 0 =="
DET=$RUNS/monai_detector/detections.csv
if [ ! -s "$DET" ]; then
  tmp=$(mktemp -d)
  if git clone -q --depth 1 --filter=blob:none --sparse -b pilot-results "https://github.com/$REPO.git" "$tmp" \
     && git -C "$tmp" sparse-checkout set "$DET_REF" && [ -s "$tmp/$DET_REF/detections.csv" ]; then
    mkdir -p "$(dirname "$DET")" && cp "$tmp/$DET_REF/detections.csv" "$DET"
    echo "using saved detections from pilot-results/$DET_REF"
  else
    echo "saved detections not found: running the detector"
    python -m scripts.eval_monai_detector --luna_root "$DATA" --subset 0 \
      --bundle_dir "$WS/models/bundles" --out "$RUNS/monai_detector"
  fi
fi

echo "== 5/5 chain filter after detector =="
python -m scripts.chain_fp_filter --detections "$DET" --weights "$RUNS/fp_filter/best.pt" \
  --meta "$PROC/meta.csv" --npy_dir "$PROC/npy" --luna_root "$DATA" --subset 0 --out "$RUNS/fp_chain"

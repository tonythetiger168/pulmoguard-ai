#!/usr/bin/env bash
# runpod_pilot.sh — 在單一 RunPod pod 上一次跑完最小試跑，跑完自動關機（停止計費）
#   1) 下載 LUNA16 subset0（6.8 GB）  2) 前處理  3) MONAI 預訓練偵測器 FROC/CPM
#   4) 假陽性削減 baseline 訓練（AMP、早停、可續跑）  5) 結果推回 GitHub → 關 pod
# 用法（pod 內，repo 根目錄）：
#   bash scripts/runpod_pilot.sh            # 預設資料放 /workspace（pod volume，重啟不遺失）
#   KEEP_POD=1 bash scripts/runpod_pilot.sh # 跑完不關機
# 自動存回結果：在 RunPod 設定環境變數 GH_TOKEN（fine-grained PAT，只授權本 repo 的 Contents: write）
#   → 結果推到 pilot-results 分支後「刪除」pod（GPU 與磁碟都不再計費）
#   沒設 GH_TOKEN 或推送失敗 → 只「停止」pod，結果留在 /workspace volume
#   KEEP_VOLUME=1 → 推送成功也只停止，保留已下載的資料供下次使用
set -euo pipefail
WS=${WS:-/workspace}
DATA=$WS/data/luna16
PROC=$WS/data/processed
RUNS=$WS/runs
LOG=$RUNS/pilot_$(date +%Y%m%d_%H%M).log
mkdir -p "$RUNS"
exec > >(tee -a "$LOG") 2>&1
t0=$(date +%s)

save_results() {
  # 只存小檔（指標、偵測清單、log），不存 CT 或模型權重
  OUT_DIR="results/pilot_$(date +%Y%m%d_%H%M)"
  mkdir -p "$OUT_DIR"
  cp "$RUNS/monai_detector/froc.json" "$RUNS/monai_detector/detections.csv" "$OUT_DIR/" 2>/dev/null || true
  cp "$RUNS/monai_detector/error.txt" "$OUT_DIR/detector_error.txt" 2>/dev/null || true
  cp "$RUNS/baseline/metrics.json" "$OUT_DIR/" 2>/dev/null || true
  cp "$LOG" "$OUT_DIR/run.log" 2>/dev/null || true
  nvidia-smi --query-gpu=name --format=csv,noheader > "$OUT_DIR/gpu.txt" 2>/dev/null || true
  if [ -z "${GH_TOKEN:-}" ]; then
    echo "GH_TOKEN not set: results kept on the volume only ($OUT_DIR)"
    return 1
  fi
  # 推到獨立分支 pilot-results（不動 main 與 PR 分支）；token 走 header，不寫進 URL 或 log
  local auth repo=${RESULTS_REPO:-tonythetiger168/pulmoguard-ai}
  local remote=${RESULTS_REMOTE:-https://github.com/$repo.git}
  auth=$(printf 'x-access-token:%s' "$GH_TOKEN" | base64 -w0)
  local tmp; tmp=$(mktemp -d)
  local G=(git -c "http.https://github.com/.extraheader=AUTHORIZATION: basic $auth")
  if "${G[@]}" clone -q --depth 1 --branch pilot-results "$remote" "$tmp" 2>/dev/null; then :
  else
    "${G[@]}" clone -q --depth 1 "$remote" "$tmp" && git -C "$tmp" checkout -q --orphan pilot-results && git -C "$tmp" rm -rqf .
  fi
  mkdir -p "$tmp/$OUT_DIR" && cp -r "$OUT_DIR/." "$tmp/$OUT_DIR/"
  git -C "$tmp" add "$OUT_DIR"
  git -C "$tmp" -c user.name="runpod-pilot" -c user.email="runpod-pilot@users.noreply.github.com" \
    commit -qm "pilot results $(basename "$OUT_DIR") on $(cat "$OUT_DIR/gpu.txt" 2>/dev/null)"
  (cd "$tmp" && "${G[@]}" push -q origin pilot-results) && \
    echo "results pushed: https://github.com/$repo/tree/pilot-results/$OUT_DIR"
}

finish() {
  echo "elapsed: $(( ($(date +%s) - t0) / 60 )) min. log: $LOG"
  local saved=0
  save_results && saved=1
  [ "${KEEP_POD:-0}" = "1" ] && return
  [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null || return
  if [ "$saved" = "1" ] && [ "${KEEP_VOLUME:-0}" != "1" ]; then
    echo "results are on GitHub: removing pod $RUNPOD_POD_ID (no more GPU or disk charges)"
    runpodctl remove pod "$RUNPOD_POD_ID"
  else
    echo "stopping pod $RUNPOD_POD_ID (GPU billing stops; volume kept so results are not lost)"
    runpodctl stop pod "$RUNPOD_POD_ID"
  fi
}
trap finish EXIT   # 不論成功或失敗：存結果 → 關機，避免閒置計費

pip install -q -r requirements.txt
nvidia-smi --query-gpu=name,memory.total --format=csv || true

echo "== 1/4 download subset0 =="
bash scripts/download_luna16.sh "$DATA" 0

echo "== 2/4 preprocess =="
python -m src.preprocess.pipeline --luna_root "$DATA" --out "$PROC" --subsets 0   # workers 依容器記憶體自動決定（nproc 在容器內會回報整台主機核心數）

echo "== 3/4 pretrained MONAI detector on subset0 (held-out fold for these weights) =="
python -m scripts.eval_monai_detector --luna_root "$DATA" --subset 0 \
  --bundle_dir "$WS/models/bundles" --out "$RUNS/monai_detector" || echo "DETECTOR STEP FAILED (see detector_error.txt); continuing with baseline"

echo "== 4/4 FP-reduction baseline =="
python -m scripts.train_baseline --meta "$PROC/meta.csv" --npy_dir "$PROC/npy" \
  --candidates "$DATA/candidates_V2.csv" --out "$RUNS/baseline" --resume \
  --num_workers 4

echo "== results =="
cat "$RUNS/monai_detector/froc.json" 2>/dev/null || true
python -c "import json;m=json.load(open('$RUNS/baseline/metrics.json'));m.pop('history');print(m)"

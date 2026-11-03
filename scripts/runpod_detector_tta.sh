#!/usr/bin/env bash
# runpod_detector_tta.sh — MONAI 預訓練偵測器 + 翻轉 TTA，LUNA16 subset0 的 CPM 比較；結果推回 GitHub 後刪除 pod
#   0) 先確認 GitHub 推得上去  1) 下載 subset0（6.8 GB）  2) 4 次推論（none/x/y/xy）+ 合併  3) 推結果 → 刪 pod
# 用法（pod 內，repo 根目錄）：bash scripts/runpod_detector_tta.sh
#   FLIPS=none,x,y,z,xy bash scripts/runpod_detector_tta.sh
# 需要 GH_TOKEN（本 repo Contents: Read and write）
set -euo pipefail
WS=${WS:-/workspace}
DATA=$WS/data/luna16
RUNS=$WS/runs/detector_tta
STAMP=$(date +%Y%m%d_%H%M)
LOG=$WS/runs/detector_tta_$STAMP.log
mkdir -p "$RUNS"
exec > >(tee -a "$LOG") 2>&1
t0=$(date +%s)
REPO=${RESULTS_REPO:-tonythetiger168/pulmoguard-ai}
REMOTE=${RESULTS_REMOTE:-https://github.com/$REPO.git}

git_auth() {  # token 走 header，不寫進 URL 或 log
  local auth; auth=$(printf 'x-access-token:%s' "$GH_TOKEN" | base64 -w0)
  git -c "http.https://github.com/.extraheader=AUTHORIZATION: basic $auth" "$@"
}

clone_results() {
  git_auth clone -q --depth 1 --branch pilot-results "$REMOTE" "$1"
}

preflight() {
  [ -n "${GH_TOKEN:-}" ] || { echo "PREFLIGHT FAILED: GH_TOKEN not set"; return 1; }
  local tmp; tmp=$(mktemp -d)
  clone_results "$tmp" || { echo "PREFLIGHT FAILED: clone"; return 1; }
  git -C "$tmp" -c user.name=runpod -c user.email=runpod@users.noreply.github.com commit -q --allow-empty -m preflight
  (cd "$tmp" && git_auth push --dry-run -q origin HEAD:pilot-results) || {
    echo "PREFLIGHT FAILED: token cannot push (needs Contents: Read and write on $REPO)"; return 1; }
  echo "preflight ok: GitHub push works"
}

save_results() {
  OUT_DIR="results/detector_tta_$STAMP"
  mkdir -p "$OUT_DIR"
  cp "$RUNS"/summary.json "$RUNS"/merged_*.csv "$OUT_DIR/" 2>/dev/null || true
  for d in "$RUNS"/flip_*; do
    [ -d "$d" ] || continue
    mkdir -p "$OUT_DIR/$(basename "$d")" && cp "$d"/detections.csv "$OUT_DIR/$(basename "$d")/" 2>/dev/null || true
  done
  cp "$LOG" "$OUT_DIR/run.log" 2>/dev/null || true
  nvidia-smi --query-gpu=name --format=csv,noheader > "$OUT_DIR/gpu.txt" 2>/dev/null || true
  [ -n "${GH_TOKEN:-}" ] || return 1
  local tmp; tmp=$(mktemp -d)
  clone_results "$tmp" || return 1
  mkdir -p "$tmp/$OUT_DIR" && cp -r "$OUT_DIR/." "$tmp/$OUT_DIR/"
  git -C "$tmp" add "$OUT_DIR"
  git -C "$tmp" -c user.name=runpod-tta -c user.email=runpod-tta@users.noreply.github.com \
    commit -qm "detector TTA results $STAMP on $(cat "$OUT_DIR/gpu.txt" 2>/dev/null)"
  for i in 1 2 3 4; do
    if (cd "$tmp" && git_auth pull -q --rebase origin pilot-results 2>/dev/null; git_auth push -q origin HEAD:pilot-results); then
      echo "results pushed: https://github.com/$REPO/tree/pilot-results/$OUT_DIR"; return 0
    fi
    sleep $((2 ** i))
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
    echo "results NOT pushed: stopping pod $RUNPOD_POD_ID so the volume and log survive"; runpodctl stop pod "$RUNPOD_POD_ID"
  fi
}
trap finish EXIT

command -v unzip >/dev/null || (apt-get update -qq && apt-get install -y -qq unzip) || true
preflight
pip install -q -r requirements.txt
nvidia-smi --query-gpu=name,memory.total --format=csv || true

echo "== 1/2 download LUNA16 subset0 =="
bash scripts/download_luna16.sh "$DATA" 0

echo "== 2/2 detector with flip TTA =="
python -m scripts.detector_tta --luna_root "$DATA" --subset 0 --bundle_dir "$WS/models/bundles" \
  --out "$RUNS" --flips "${FLIPS:-none,x,y,xy}"

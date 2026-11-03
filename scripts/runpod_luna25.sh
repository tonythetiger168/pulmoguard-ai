#!/usr/bin/env bash
# runpod_luna25.sh — 在單一 RunPod pod 上訓練 LUNA25 惡性度模型，結果推回 GitHub 後刪除 pod
#   0) 開機先確認 GitHub 推得上去（推不上去就不花 GPU 時間，直接停）
#   1) 下載官方 nodule blocks（6.8 GB）  2) 5-fold 病人層級 CV 訓練  3) 結果推到 pilot-results 分支 → 刪 pod
# 用法（pod 內，repo 根目錄）：
#   bash scripts/runpod_luna25.sh
#   EXTRA="--only_fold 0" bash scripts/runpod_luna25.sh   # 只跑一個 fold，先看趨勢
#   KEEP_POD=1 bash scripts/runpod_luna25.sh              # 跑完不刪 pod
# 需要環境變數 GH_TOKEN（fine-grained PAT，本 repo Contents: Read and write）
# 推回 pilot-results 的只有小檔（summary、各 fold metrics、OOF 預測、log）；
# 模型權重（每 fold fp16 約 66 MB）上傳到 GitHub Release luna25-<時間>，失敗則推到 luna25-weights 分支
set -euo pipefail
WS=${WS:-/workspace}
DATA=$WS/data/luna25
RUNS=$WS/runs/luna25
STAMP=$(date +%Y%m%d_%H%M)
LOG=$WS/runs/luna25_$STAMP.log
mkdir -p "$RUNS"
exec > >(tee -a "$LOG") 2>&1
t0=$(date +%s)
REPO=${RESULTS_REPO:-tonythetiger168/pulmoguard-ai}
REMOTE=${RESULTS_REMOTE:-https://github.com/$REPO.git}

git_auth() {  # token 走 header，不寫進 URL 或 log
  local auth; auth=$(printf 'x-access-token:%s' "$GH_TOKEN" | base64 -w0)
  git -c "http.https://github.com/.extraheader=AUTHORIZATION: basic $auth" "$@"
}

clone_results() {  # $1 = 目的資料夾；pilot-results 分支不存在就建立 orphan 分支
  if git_auth clone -q --depth 1 --branch pilot-results "$REMOTE" "$1" 2>/dev/null; then :
  else
    git_auth clone -q --depth 1 "$REMOTE" "$1" && git -C "$1" checkout -q --orphan pilot-results && git -C "$1" rm -rqf .
  fi
}

preflight() {
  [ -n "${GH_TOKEN:-}" ] || { echo "PREFLIGHT FAILED: GH_TOKEN not set"; return 1; }
  local tmp; tmp=$(mktemp -d)
  clone_results "$tmp" || { echo "PREFLIGHT FAILED: clone"; return 1; }
  git -C "$tmp" -c user.name=runpod -c user.email=runpod@users.noreply.github.com \
    commit -q --allow-empty -m preflight
  (cd "$tmp" && git_auth push --dry-run -q origin HEAD:pilot-results) || {
    echo "PREFLIGHT FAILED: token cannot push (needs Contents: Read and write on $REPO)"; return 1; }
  echo "preflight ok: GitHub push works"
}

save_results() {
  OUT_DIR="results/luna25_$STAMP"
  mkdir -p "$OUT_DIR"
  cp "$RUNS"/summary.json "$RUNS"/oof_predictions.csv "$RUNS"/folds.csv "$OUT_DIR/" 2>/dev/null || true
  for d in "$RUNS"/fold*; do
    [ -d "$d" ] || continue
    mkdir -p "$OUT_DIR/$(basename "$d")"
    cp "$d"/metrics.json "$d"/test_predictions.csv "$OUT_DIR/$(basename "$d")/" 2>/dev/null || true
  done
  cp "$LOG" "$OUT_DIR/run.log" 2>/dev/null || true
  nvidia-smi --query-gpu=name --format=csv,noheader > "$OUT_DIR/gpu.txt" 2>/dev/null || true
  [ -n "${GH_TOKEN:-}" ] || { echo "GH_TOKEN not set: results kept on the volume only"; return 1; }
  local tmp; tmp=$(mktemp -d)
  clone_results "$tmp" || return 1
  mkdir -p "$tmp/$OUT_DIR" && cp -r "$OUT_DIR/." "$tmp/$OUT_DIR/"
  git -C "$tmp" add "$OUT_DIR"
  git -C "$tmp" -c user.name="runpod-luna25" -c user.email="runpod-luna25@users.noreply.github.com" \
    commit -qm "luna25 results $STAMP on $(cat "$OUT_DIR/gpu.txt" 2>/dev/null)"
  for i in 1 2 3 4; do
    if (cd "$tmp" && git_auth pull -q --rebase origin pilot-results 2>/dev/null; git_auth push -q origin HEAD:pilot-results); then
      echo "results pushed: https://github.com/$REPO/tree/pilot-results/$OUT_DIR"; return 0
    fi
    sleep $((2 ** i))
  done
  return 1
}

save_weights() {
  # 每個 fold 的 best.pt 轉成 fp16（約 66 MB）上傳到 GitHub Release；失敗就推到 luna25-weights 分支（單檔 < 100 MB）
  ls "$RUNS"/fold*/best.pt >/dev/null 2>&1 || { echo "no weights to save"; return 0; }
  [ -n "${GH_TOKEN:-}" ] || { echo "GH_TOKEN not set: weights not saved"; return 1; }
  local W=$WS/weights; mkdir -p "$W"
  python - "$RUNS" "$W" <<'PY' || return 1
import glob, os, sys, torch
runs, out = sys.argv[1], sys.argv[2]
for p in sorted(glob.glob(f"{runs}/fold*/best.pt")):
    k = os.path.basename(os.path.dirname(p))
    sd = {n: (t.half() if t.is_floating_point() else t) for n, t in torch.load(p, map_location="cpu").items()}
    torch.save(sd, f"{out}/luna25_resnet18_{k}_fp16.pt")
    print("exported", k, round(os.path.getsize(f"{out}/luna25_resnet18_{k}_fp16.pt") / 1e6, 1), "MB")
PY
  cp "$RUNS/summary.json" "$W/" 2>/dev/null || true
  local tag="luna25-$STAMP" sha api="https://api.github.com/repos/$REPO" rel id ok=1
  sha=$(git rev-parse HEAD 2>/dev/null || echo main)
  local body; body=$(python -c "import json;s=json.load(open('$RUNS/summary.json'));print(json.dumps({'tag_name':'$tag','target_commitish':'$sha','name':'LUNA25 malignancy ResNet-18 ($STAMP)','prerelease':True,'body':f\"5-fold patient-level CV, OOF AUC {s['oof_auc']:.3f} (95% CI {s['oof_auc_95ci'][0]:.3f}-{s['oof_auc_95ci'][1]:.3f}). fp16 state_dicts for monai resnet18(spatial_dims=3, n_input_channels=1, num_classes=1); input 64^3 patch, 50 mm FOV, HU -1000..400 scaled to 0..1. Labels CC BY-NC 4.0: research use only.\"}))" 2>/dev/null) \
    || body="{\"tag_name\":\"$tag\",\"target_commitish\":\"$sha\",\"prerelease\":true}"
  rel=$(curl -sS -X POST -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" "$api/releases" -d "$body")
  id=$(echo "$rel" | python -c "import json,sys;print(json.load(sys.stdin).get('id',''))" 2>/dev/null)
  if [ -n "$id" ]; then
    for f in "$W"/*; do
      curl -sS -f -X POST -H "Authorization: Bearer $GH_TOKEN" -H "Content-Type: application/octet-stream" \
        --data-binary @"$f" "https://uploads.github.com/repos/$REPO/releases/$id/assets?name=$(basename "$f")" >/dev/null \
        && echo "uploaded $(basename "$f")" || ok=0
    done
    [ "$ok" = 1 ] && { echo "weights uploaded: https://github.com/$REPO/releases/tag/$tag"; return 0; }
  else
    echo "release create failed: $(echo "$rel" | head -c 300)"
  fi
  echo "falling back to branch luna25-weights"
  local tmp; tmp=$(mktemp -d)
  git_auth clone -q --depth 1 "$REMOTE" "$tmp" && git -C "$tmp" checkout -q --orphan luna25-weights && git -C "$tmp" rm -rqf . || return 1
  mkdir -p "$tmp/$tag" && cp "$W"/* "$tmp/$tag/"
  git -C "$tmp" add "$tag" && git -C "$tmp" -c user.name=runpod-luna25 -c user.email=runpod-luna25@users.noreply.github.com commit -qm "luna25 weights $tag"
  (cd "$tmp" && git_auth push -q origin HEAD:luna25-weights 2>/dev/null || { git_auth fetch -q --depth 1 origin luna25-weights && git reset -q --soft FETCH_HEAD && git commit -qm "luna25 weights $tag" && git_auth push -q origin HEAD:luna25-weights; }) \
    && echo "weights uploaded: https://github.com/$REPO/tree/luna25-weights/$tag"
}

finish() {
  echo "elapsed: $(( ($(date +%s) - t0) / 60 )) min. log: $LOG"
  local saved=0
  save_weights || echo "WEIGHTS NOT SAVED"
  save_results && saved=1
  [ "${KEEP_POD:-0}" = "1" ] && return
  [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null || return
  if [ "$saved" = "1" ]; then
    echo "results are on GitHub: removing pod $RUNPOD_POD_ID"
    runpodctl remove pod "$RUNPOD_POD_ID"
  else
    echo "results NOT pushed: stopping pod $RUNPOD_POD_ID so the volume and log survive"
    runpodctl stop pod "$RUNPOD_POD_ID"
  fi
}
trap finish EXIT   # 不論成功或失敗：存結果 → 關機，避免閒置計費

command -v unzip >/dev/null || (apt-get update -qq && apt-get install -y -qq unzip) || true
preflight
pip install -q -r requirements.txt
nvidia-smi --query-gpu=name,memory.total --format=csv || true

echo "== 1/2 download LUNA25 nodule blocks =="
bash scripts/download_luna25.sh "$DATA"

echo "== 2/2 train malignancy model (patient-level 5-fold CV) =="
# shellcheck disable=SC2086
python -m scripts.train_luna25 --csv "$DATA/LUNA25_Public_Training_Development_Data.csv" \
  --blocks "$DATA/nodule_blocks" --out "$RUNS" ${EXTRA:-}

echo "== results =="
python -c "import json;s=json.load(open('$RUNS/summary.json'));print({k:s[k] for k in ('oof_auc','oof_auc_95ci','fold_auc_mean','wall_hours')})"

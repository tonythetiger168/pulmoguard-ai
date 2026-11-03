#!/usr/bin/env bash
# runpod_nlst_eval.sh — NLST 病人層級驗證（LUNA25 未見過的 NLST 病人），結果推回 GitHub 後刪除 pod
#   0) GitHub push 預檢
#   1) 下載 LUNA25 nodule blocks + CSV → 挑 NLST 世代（排除 LUNA25 病人）
#   2) 背景：從 IDC 分批下載 NLST T0 CT → 1 mm 壓縮 .mhd（轉完即刪 DICOM）
#      前景：重訓 LUNA25 5-fold（上一次沒存權重），權重存成 GitHub Release
#   3) MONAI 偵測器 → LUNA25 集成 → 病人分數、AUC
# 用法（pod 內，repo 根目錄）：bash scripts/runpod_nlst_eval.sh      LIMIT=20 → 小量試跑
# 需要環境變數 GH_TOKEN（fine-grained PAT，本 repo Contents: Read and write）
set -euo pipefail
export PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
pipi() { for i in 1 2 3; do pip install -q "$@" && return 0; echo "pip install failed (try $i), retrying"; sleep $((30 * i)); done; return 1; }
WS=${WS:-/workspace}
STAMP=$(date +%Y%m%d_%H%M)
LOG=$WS/runs/nlst_$STAMP.log
L25=$WS/data/luna25
RUNS=$WS/runs/luna25          # save_weights 用這個變數
NRUN=$WS/runs/nlst
MHD=$WS/data/nlst/mhd
mkdir -p "$RUNS" "$NRUN"
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
  OUT_DIR="results/nlst_$STAMP"
  mkdir -p "$OUT_DIR/luna25"
  cp "$NRUN"/summary.json "$NRUN"/patient_scores.csv "$NRUN"/candidate_scores.csv "$NRUN"/cohort.csv "$OUT_DIR/" 2>/dev/null || true
  gzip -c "$NRUN/detections.csv" > "$OUT_DIR/detections.csv.gz" 2>/dev/null || true
  cp "$MHD/prepare_log.jsonl" "$OUT_DIR/" 2>/dev/null || true
  cp "$RUNS"/summary.json "$RUNS"/oof_predictions.csv "$OUT_DIR/luna25/" 2>/dev/null || true
  cp "$LOG" "$OUT_DIR/run.log" 2>/dev/null || true
  [ -f "$WS/runs/nlst_prepare.log" ] && cp "$WS/runs/nlst_prepare.log" "$OUT_DIR/prepare.log"
  nvidia-smi --query-gpu=name --format=csv,noheader > "$OUT_DIR/gpu.txt" 2>/dev/null || true
  [ -n "${GH_TOKEN:-}" ] || { echo "GH_TOKEN not set"; return 1; }
  local tmp; tmp=$(mktemp -d)
  clone_results "$tmp" || return 1
  mkdir -p "$tmp/$OUT_DIR" && cp -r "$OUT_DIR/." "$tmp/$OUT_DIR/"
  git -C "$tmp" add "$OUT_DIR"
  git -C "$tmp" -c user.name="runpod-nlst" -c user.email="runpod-nlst@users.noreply.github.com" \
    commit -qm "nlst validation $STAMP on $(cat "$OUT_DIR/gpu.txt" 2>/dev/null)"
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
  local tag="luna25-nlst-$STAMP" sha api="https://api.github.com/repos/$REPO" rel id ok=1
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
  local wb=luna25-weights
  echo "falling back to branch $wb"
  local tmp; tmp=$(mktemp -d)
  git_auth clone -q --depth 1 "$REMOTE" "$tmp" && git -C "$tmp" checkout -q --orphan "$wb" && git -C "$tmp" rm -rqf . || return 1
  mkdir -p "$tmp/$tag" && cp "$W"/* "$tmp/$tag/"
  git -C "$tmp" add "$tag" && git -C "$tmp" -c user.name=runpod-luna25 -c user.email=runpod-luna25@users.noreply.github.com commit -qm "luna25 weights $tag"
  local ref="HEAD:$wb"
  if ! (cd "$tmp" && git_auth push -q origin "$ref" 2>/dev/null); then   # 分支已存在：接在它後面再推
    (cd "$tmp" && git_auth fetch -q --depth 1 origin "$wb" && git reset -q --soft FETCH_HEAD \
      && git commit -qm "luna25 weights $tag" && git_auth push -q origin "$ref") || return 1
  fi
  echo "weights uploaded: https://github.com/$REPO/tree/$wb/$tag"
}

finish() {
  local rc=$?
  [ -n "${PREP_PID:-}" ] && kill "$PREP_PID" 2>/dev/null || true
  echo "exit code $rc; elapsed: $(( ($(date +%s) - t0) / 60 )) min. log: $LOG"
  [ "${SKIP_LUNA25_TRAIN:-0}" = "1" ] || save_weights || echo "WEIGHTS NOT SAVED"
  save_results || echo "results NOT pushed"
}
trap finish EXIT

command -v unzip >/dev/null || (apt-get update -qq && apt-get install -y -qq unzip wget) || true
preflight
pipi -r requirements.txt
pipi idc-index
python -c "import torch;print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
NCPU=$(python -c "from src.preprocess.pipeline import _container_limits as c;print(c()[0])")
NW=$(( NCPU > 8 ? 6 : (NCPU > 2 ? NCPU - 2 : 1) ))
echo "cpus=$NCPU nlst_workers=$NW"

fetch_weights() {  # LUNA25_WEIGHTS_TAG=luna25-YYYYMMDD_HHMM → 直接用另一次訓練存好的 fold 權重，不重訓
  local api="https://api.github.com/repos/$REPO/releases/tags/$LUNA25_WEIGHTS_TAG"
  curl -sS -H "Authorization: Bearer $GH_TOKEN" "$api" | python -c "
import json,sys
for a in json.load(sys.stdin).get('assets',[]):
    if a['name'].endswith('_fp16.pt'): print(a['url'], a['name'])" | while read -r url name; do
    k=$(echo "$name" | grep -oE 'fold[0-9]+'); mkdir -p "$RUNS/$k"
    curl -sS -fL -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/octet-stream" -o "$RUNS/$k/best.pt" "$url"
    echo "weights $name -> $RUNS/$k/best.pt"
  done
  [ "$(ls "$RUNS"/fold*/best.pt 2>/dev/null | wc -l)" -ge 5 ]
}

echo "== 1/4 LUNA25 data + NLST cohort =="
random_weights() {  # 只給冒煙測試用：隨機初始化的 5 個 fold 模型，驗證整條管線能跑通（分數無意義）
  python - "$RUNS" <<'PY'
import os, sys, torch
from monai.networks.nets import resnet18
for k in range(5):
    os.makedirs(f"{sys.argv[1]}/fold{k}", exist_ok=True)
    torch.save(resnet18(spatial_dims=3, n_input_channels=1, num_classes=1).state_dict(), f"{sys.argv[1]}/fold{k}/best.pt")
PY
}

if [ "${SMOKE_RANDOM_WEIGHTS:-0}" = "1" ] && random_weights || { [ -n "${LUNA25_WEIGHTS_TAG:-}" ] && fetch_weights; }; then
  SKIP_LUNA25_TRAIN=1
  mkdir -p "$L25" && wget -q -O "$L25/LUNA25_Public_Training_Development_Data.csv" \
    "https://zenodo.org/records/14673658/files/LUNA25_Public_Training_Development_Data.csv?download=1"
else
  SKIP_LUNA25_TRAIN=0
  bash scripts/download_luna25.sh "$L25"
fi
python -m scripts.nlst_cohort --luna25_csv "$L25/LUNA25_Public_Training_Development_Data.csv" --out "$NRUN/cohort.csv"
if [ -n "${LIMIT:-}" ]; then
  python - "$NRUN/cohort.csv" "$LIMIT" <<'PY'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1], dtype=str)
n = int(sys.argv[2]) // 2
pd.concat([df[df.label == "1"].head(n), df[df.label == "0"].head(n)]).to_csv(sys.argv[1], index=False)
PY
fi

echo "== 2/4 NLST download+convert (background) and LUNA25 5-fold training =="
python -m scripts.nlst_prepare --cohort "$NRUN/cohort.csv" --out "$MHD" --workers "$NW" \
  > "$WS/runs/nlst_prepare.log" 2>&1 & PREP_PID=$!
if [ "$SKIP_LUNA25_TRAIN" = "0" ]; then
# shellcheck disable=SC2086
python -m scripts.train_luna25 --csv "$L25/LUNA25_Public_Training_Development_Data.csv" \
  --blocks "$L25/nodule_blocks" --out "$RUNS" ${LUNA25_EXTRA:-}
fi
echo "LUNA25 done; waiting for NLST preparation"
wait "$PREP_PID"; PREP_PID=""
tail -3 "$WS/runs/nlst_prepare.log"
rm -rf "$L25/nodule_blocks" 2>/dev/null || true

echo "== 3/4 detector + malignancy on NLST =="
python -m scripts.nlst_validate --cohort "$NRUN/cohort.csv" --mhd_dir "$MHD" --luna25_runs "$RUNS" \
  --bundle_dir "$WS/models/bundles" --out "$NRUN"

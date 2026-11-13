#!/usr/bin/env bash
# runpod_nlst_long.sh — NLST 縱向（T0/T1/T2）結節特徵抽取：偵測 → LIDC 分割 → LUNA25 特徵 → 前後次對位
# 結果每 3 小時與結束時推回 GitHub pilot-results/results/nlst_long_<STAMP>/，推送成功後自刪 pod
# 用法（pod 內，repo 根目錄）：
#   LIMIT_PATIENTS=50 bash scripts/runpod_nlst_long.sh        # 試跑
#   bash scripts/runpod_nlst_long.sh                          # 全量（2,695 人）
#   RESUME_DIR=results/nlst_long_YYYYMMDD_HHMM bash ...        # 從上次推回的進度續跑
#   SHARD=0 NSHARD=2 bash ...                                  # 兩個 pod 各跑一半病人
# 需要 GH_TOKEN（本 repo Contents: Read and write）
set -euo pipefail
export PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
pipi() { for i in 1 2 3; do pip install -q "$@" && return 0; echo "pip install failed (try $i), retrying"; sleep $((30 * i)); done; return 1; }
WS=${WS:-/workspace}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M)}
LOG=$WS/runs/nlst_long_$STAMP.log
L25=$WS/data/luna25
RUNS=$WS/runs/luna25
NRUN=$WS/runs/nlst_long
MHD=$WS/data/nlst/mhd
SEG=$WS/models/lidc_roi
LUNA25_WEIGHTS_TAG=${LUNA25_WEIGHTS_TAG:-luna25-20261103_0849}
SEG_TAG=${SEG_TAG:-lidc-nnunet-20261103_0855}
OUT_DIR=results/nlst_long_$STAMP
mkdir -p "$RUNS" "$NRUN" "$SEG/fold_0" "$WS/runs"
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

save_results() {  # $1 = 說明文字
  local stage=$WS/stage/$OUT_DIR; rm -rf "$stage"; mkdir -p "$stage/embeddings"
  cp "$NRUN"/cohort.csv "$NRUN"/nodules.csv "$NRUN"/transforms.csv "$NRUN"/done.txt "$NRUN"/detector_failed.txt "$stage/" 2>/dev/null || true
  cp "$NRUN"/embeddings/*.npy "$stage/embeddings/" 2>/dev/null || true
  cp "$MHD/prepare_log.jsonl" "$stage/" 2>/dev/null || true
  cp "$LOG" "$stage/run.log" 2>/dev/null || true
  nvidia-smi --query-gpu=name --format=csv,noheader > "$stage/gpu.txt" 2>/dev/null || true
  echo "{\"elapsed_min\": $(( ($(date +%s) - t0) / 60 )), \"note\": \"$1\"}" > "$stage/status.json"
  [ -n "${GH_TOKEN:-}" ] || { echo "GH_TOKEN not set"; return 1; }
  local tmp; tmp=$(mktemp -d)
  clone_results "$tmp" || return 1
  mkdir -p "$tmp/$OUT_DIR" && cp -r "$stage/." "$tmp/$OUT_DIR/"
  git -C "$tmp" add "$OUT_DIR"
  git -C "$tmp" -c user.name="runpod-nlst-long" -c user.email="runpod-nlst-long@users.noreply.github.com" \
    commit -qm "nlst longitudinal $STAMP: $1" || true
  for i in 1 2 3 4; do
    if (cd "$tmp" && git_auth pull -q --rebase origin pilot-results 2>/dev/null; git_auth push -q origin HEAD:pilot-results); then
      echo "results pushed ($1): https://github.com/$REPO/tree/pilot-results/$OUT_DIR"; rm -rf "$tmp"; return 0
    fi
    sleep $((2 ** i))
  done
  rm -rf "$tmp"; return 1
}

remove_pod() {
  [ "${KEEP_POD:-0}" = "1" ] && return 0
  [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null || return 0
  if grep -q "results pushed (final" "$LOG"; then runpodctl remove pod "$RUNPOD_POD_ID"; else runpodctl stop pod "$RUNPOD_POD_ID"; fi
}

finish() {
  local rc=$?
  [ -n "${PREP_PID:-}" ] && kill "$PREP_PID" 2>/dev/null || true
  [ -n "${PUSH_PID:-}" ] && kill "$PUSH_PID" 2>/dev/null || true
  echo "exit code $rc; elapsed: $(( ($(date +%s) - t0) / 60 )) min. log: $LOG"
  save_results "final rc=$rc" || echo "results NOT pushed"
  remove_pod
}
trap finish EXIT

preflight
pipi -r requirements.txt
pipi "nnunetv2>=2.6,<2.9" idc-index
if python -c "import torch,sys;sys.exit(0 if 'dev' in torch.__version__ else 1)"; then
  pipi "torch==2.7.1" "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cu128
fi
export nnUNet_compile=f nnUNet_raw=$WS/nnunet/raw nnUNet_preprocessed=$WS/nnunet/pre nnUNet_results=$WS/nnunet/res
python -c "import torch;print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
NCPU=$(python -c "from src.preprocess.pipeline import _container_limits as c;print(c()[0])")
NW=$(( NCPU > 8 ? 6 : (NCPU > 2 ? NCPU - 2 : 1) ))
NREG=$(( NCPU > 8 ? 4 : 2 ))
echo "cpus=$NCPU prepare_workers=$NW reg_workers=$NREG"

echo "== 1/4 weights (LUNA25 $LUNA25_WEIGHTS_TAG, LIDC ROI $SEG_TAG) =="
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

fetch_weights
release_asset() {  # $1 tag, $2 asset name, $3 destination
  local url; url=$(curl -sS -H "Authorization: Bearer $GH_TOKEN" "https://api.github.com/repos/$REPO/releases/tags/$1" \
    | python -c "import json,sys;print(next(a['url'] for a in json.load(sys.stdin)['assets'] if a['name']=='$2'))")
  curl -sS -fL -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/octet-stream" -o "$3" "$url"
}
release_asset "$SEG_TAG" checkpoint_final.pth "$SEG/fold_0/checkpoint_final.pth"
release_asset "$SEG_TAG" nnUNetPlans.json "$SEG/plans.json"
release_asset "$SEG_TAG" dataset.json "$SEG/dataset.json"
ls -la "$SEG" "$SEG/fold_0"

echo "== 2/4 cohort =="
mkdir -p "$L25" && wget -q -O "$L25/LUNA25_Public_Training_Development_Data.csv" \
  "https://zenodo.org/records/14673658/files/LUNA25_Public_Training_Development_Data.csv?download=1" || true
L25CSV=()
[ -s "$L25/LUNA25_Public_Training_Development_Data.csv" ] && L25CSV=(--luna25_csv "$L25/LUNA25_Public_Training_Development_Data.csv")
if [ -n "${RESUME_DIR:-}" ]; then
  tmp=$(mktemp -d); clone_results "$tmp"
  cp -r "$tmp/$RESUME_DIR/." "$NRUN/" && rm -f "$NRUN/run.log" "$NRUN/status.json" "$NRUN/prepare_log.jsonl"
  echo "resumed from $RESUME_DIR: $(wc -l < "$NRUN/done.txt") patients done"; rm -rf "$tmp"
else
  python -m scripts.nlst_longitudinal_cohort "${L25CSV[@]}" --out "$NRUN/cohort.csv" \
    ${LIMIT_PATIENTS:+--max_patients "$LIMIT_PATIENTS"}
fi

echo "== 3/4 NLST download+convert (background) =="
python - "$NRUN/cohort.csv" "$NRUN/done.txt" "$NRUN/todo_cohort.csv" <<'PY'
import os, sys, pandas as pd
c = pd.read_csv(sys.argv[1], dtype=str)
shard, nshard = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
if nshard > 1:  # 多個 pod 平行：依病人排序後取 i::n
    pts = sorted(c.PatientID.unique())[shard::nshard]
    c = c[c.PatientID.isin(pts)]
    c.to_csv(sys.argv[1], index=False)
    print(f"shard {shard}/{nshard}: {len(pts)} patients")
done = set(open(sys.argv[2]).read().split()) if os.path.exists(sys.argv[2]) else set()
c[~c.PatientID.isin(done)].to_csv(sys.argv[3], index=False)
print(f"{(~c.PatientID.isin(done)).sum()} scans to prepare")
PY
python -m scripts.nlst_prepare --cohort "$NRUN/todo_cohort.csv" --out "$MHD" --workers "$NW" \
  > "$WS/runs/nlst_prepare.log" 2>&1 & PREP_PID=$!
( while sleep 10800; do save_results "progress" || true; done ) & PUSH_PID=$!

echo "== 4/4 detector + LIDC seg + LUNA25 features + registration =="
python -m scripts.nlst_longitudinal_extract --cohort "$NRUN/cohort.csv" --mhd_dir "$MHD" --luna25_runs "$RUNS" \
  --seg_model "$SEG" --bundle_dir "$WS/models/bundles" --out "$NRUN" --reg_workers "$NREG"
wait "$PREP_PID" || echo "prepare exited non-zero: $(tail -2 "$WS/runs/nlst_prepare.log")"; PREP_PID=""
echo "nodules: $(($(wc -l < "$NRUN/nodules.csv") - 1)), registrations: $(($(wc -l < "$NRUN/transforms.csv") - 1))"

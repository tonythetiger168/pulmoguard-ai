#!/usr/bin/env bash
# runpod_lidc_nnunet.sh — LIDC-IDRI 結節分割（nnU-Net v2 3d_fullres，只跑 fold 0），結果推回 GitHub 後刪除 pod
#   0) 開機先確認 GitHub 推得上去（推不上去就不花 GPU 時間）
#   1) 從 IDC 下載 1,018 份 LIDC CT（約 124 GB，s5cmd）
#   2) pylidc 共識遮罩 → nnU-Net 資料集（病人層級 5-fold 切分），轉完刪 DICOM 省空間
#   3) nnU-Net plan + preprocess → 訓練 fold 0（每 3 小時把進度推到 GitHub）
#   4) 指標、log、plans 推到 pilot-results 分支；模型權重上傳成 GitHub Release 附件 → 刪 pod
# 用法（pod 內，repo 根目錄）：
#   bash scripts/runpod_lidc_nnunet.sh
#   LIMIT=20 EPOCHS_TRAINER=nnUNetTrainer_5epochs bash scripts/runpod_lidc_nnunet.sh   # 小量試跑
# 需要環境變數 GH_TOKEN（fine-grained PAT，本 repo Contents: Read and write）
set -euo pipefail
export PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10   # 2026-11-03：PyPI 讀取逾時讓整個 run 在第 1 分鐘就結束
pipi() { for i in 1 2 3; do pip install -q "$@" && return 0; echo "pip install failed (try $i), retrying"; sleep $((30 * i)); done; return 1; }
WS=${WS:-/workspace}
DICOM=$WS/data/lidc_dicom
FULL=Dataset501_LIDCNodule   # 全肺資料集（轉檔輸出）
# MODE=roi（預設）：以結節為中心的 64 mm ROI 分割（Dataset503）；MODE=full：全肺分割（Dataset501，兩次都學不起來）
MODE=${MODE:-roi}
if [ "$MODE" = "roi" ]; then DS=Dataset503_LIDCNoduleROI; DSID=503; else DS=$FULL; DSID=501; fi
export nnUNet_raw=$WS/nnunet/raw nnUNet_preprocessed=$WS/nnunet/preprocessed nnUNet_results=$WS/nnunet/results
TRAINER=${EPOCHS_TRAINER:-nnUNetTrainer}
FOLD=${FOLD:-0}
STAMP=$(date +%Y%m%d_%H%M)
OUT_DIR=results/lidc_nnunet_$STAMP
LOG=$WS/runs/lidc_nnunet_$STAMP.log
mkdir -p "$WS/runs" "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
exec > >(tee -a "$LOG") 2>&1
t0=$(date +%s)
REPO=${RESULTS_REPO:-tonythetiger168/pulmoguard-ai}
REMOTE=${RESULTS_REMOTE:-https://github.com/$REPO.git}
FOLD_DIR=$nnUNet_results/$DS/${TRAINER}__nnUNetPlans__3d_fullres/fold_$FOLD

# 容器內 nproc 會回報整台主機的核心數 → 依 cgroup 配額與記憶體決定 worker 數
cpus() {
  local q p
  if read -r q p < /sys/fs/cgroup/cpu.max 2>/dev/null && [ "$q" != "max" ]; then echo $(( q / p ));
  else nproc; fi
}
MEM_GB=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) / 1048576 ))
if [ -r /sys/fs/cgroup/memory.max ] && [ "$(cat /sys/fs/cgroup/memory.max)" != "max" ]; then
  MEM_GB=$(( $(cat /sys/fs/cgroup/memory.max) / 1073741824 ))
fi
NCPU=$(cpus); [ "$NCPU" -ge 1 ] || NCPU=4
NP=$(( MEM_GB / 6 )); [ "$NP" -gt "$NCPU" ] && NP=$NCPU; [ "$NP" -lt 1 ] && NP=1; [ "$NP" -gt 8 ] && NP=8
export nnUNet_n_proc_DA=$(( NCPU > 12 ? 12 : (NCPU > 2 ? NCPU - 1 : 2) ))
echo "cpus=$NCPU mem=${MEM_GB}GB preprocess_workers=$NP augmentation_workers=$nnUNet_n_proc_DA"

git_auth() {  # token 走 header，不寫進 URL 或 log
  local auth; auth=$(printf 'x-access-token:%s' "$GH_TOKEN" | base64 -w0)
  git -c "http.https://github.com/.extraheader=AUTHORIZATION: basic $auth" "$@"
}
clone_results() {
  if git_auth clone -q --depth 1 --branch pilot-results "$REMOTE" "$1" 2>/dev/null; then :
  else
    git_auth clone -q --depth 1 "$REMOTE" "$1" && git -C "$1" checkout -q --orphan pilot-results && git -C "$1" rm -rqf .
  fi
}
preflight() {
  [ -n "${GH_TOKEN:-}" ] || { echo "PREFLIGHT FAILED: GH_TOKEN not set"; return 1; }
  local tmp; tmp=$(mktemp -d)
  clone_results "$tmp" || { echo "PREFLIGHT FAILED: clone"; return 1; }
  git -C "$tmp" -c user.name=runpod -c user.email=runpod@users.noreply.github.com commit -q --allow-empty -m preflight
  (cd "$tmp" && git_auth push --dry-run -q origin HEAD:pilot-results) || {
    echo "PREFLIGHT FAILED: token cannot push (needs Contents: Read and write on $REPO)"; return 1; }
  rm -rf "$tmp"; echo "preflight ok: GitHub push works"
}

collect() {  # 只收小檔
  local d=$1; mkdir -p "$d"
  cp "$nnUNet_raw/$DS/dataset.json" "$nnUNet_raw/$DS/splits_final.json" "$d/" 2>/dev/null || true
  cp "$nnUNet_raw/$FULL/conversion_log.json" "$nnUNet_raw/$DS/rois.json" "$d/" 2>/dev/null || true
  cp "$nnUNet_preprocessed/$DS/nnUNetPlans.json" "$nnUNet_preprocessed/$DS/dataset_fingerprint.json" "$d/" 2>/dev/null || true
  if [ -d "$FOLD_DIR" ]; then
    mkdir -p "$d/fold_$FOLD"
    cp "$FOLD_DIR"/training_log_*.txt "$FOLD_DIR"/progress.png "$FOLD_DIR"/debug.json "$d/fold_$FOLD/" 2>/dev/null || true
    cp "$FOLD_DIR"/validation/summary.json "$d/fold_$FOLD/validation_summary.json" 2>/dev/null || true
  fi
  [ -f "$WS/runs/lesion_metrics.json" ] && cp "$WS/runs/lesion_metrics.json" "$d/"
  cp "$LOG" "$d/run.log" 2>/dev/null || true
  nvidia-smi --query-gpu=name --format=csv,noheader > "$d/gpu.txt" 2>/dev/null || true
}
push_dir() {  # $1 = 本地資料夾 → pilot-results/$OUT_DIR ；$2 = commit 訊息
  [ -n "${GH_TOKEN:-}" ] || return 1
  local tmp; tmp=$(mktemp -d)
  clone_results "$tmp" || return 1
  mkdir -p "$tmp/$OUT_DIR" && cp -r "$1/." "$tmp/$OUT_DIR/"
  git -C "$tmp" add "$OUT_DIR"
  git -C "$tmp" -c user.name="runpod-lidc" -c user.email="runpod-lidc@users.noreply.github.com" commit -qm "$2" || true
  for i in 1 2 3 4; do
    if (cd "$tmp" && git_auth pull -q --rebase origin pilot-results 2>/dev/null; git_auth push -q origin HEAD:pilot-results); then
      rm -rf "$tmp"; return 0; fi
    sleep $((2 ** i))
  done
  rm -rf "$tmp"; return 1
}
upload_weights() {  # 權重 > 100 MB 不能進 git → GitHub Release 附件（上限 2 GB）
  local ck=$FOLD_DIR/checkpoint_final.pth
  [ -f "$ck" ] || ck=$FOLD_DIR/checkpoint_best.pth
  [ -f "$ck" ] && [ -n "${GH_TOKEN:-}" ] || return 0
  local tag="lidc-nnunet-$STAMP" rel id
  rel=$(curl -sS -X POST -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/$REPO/releases" \
    -d "{\"tag_name\":\"$tag\",\"target_commitish\":\"pilot-results\",\"name\":\"LIDC nnU-Net fold $FOLD weights ($STAMP)\",\"body\":\"nnU-Net v2 3d_fullres $TRAINER fold $FOLD. Metrics: pilot-results/$OUT_DIR\",\"prerelease\":true}")
  id=$(printf '%s' "$rel" | python3 -c "import json,sys;print(json.load(sys.stdin).get('id',''))")
  [ -n "$id" ] || { echo "release create failed: $(printf '%s' "$rel" | head -c 300)"; return 0; }
  for f in "$ck" "$nnUNet_preprocessed/$DS/nnUNetPlans.json" "$nnUNet_raw/$DS/dataset.json"; do
    curl -sS -X POST -H "Authorization: Bearer $GH_TOKEN" -H "Content-Type: application/octet-stream" \
      --data-binary @"$f" "https://uploads.github.com/repos/$REPO/releases/$id/assets?name=$(basename "$f")" \
      -o /dev/null -w "uploaded $(basename "$f"): HTTP %{http_code}\n" || true
  done
  echo "weights: https://github.com/$REPO/releases/tag/$tag"
}

progress_loop() {  # 每 3 小時把進度推上 GitHub，cloud session 才看得到（*.proxy.runpod.net 連不到）
  while sleep "${PROGRESS_EVERY:-10800}"; do
    local d; d=$(mktemp -d); collect "$d"
    push_dir "$d" "lidc progress $STAMP ($(( ($(date +%s) - t0) / 3600 )) h)" && echo "progress pushed" || echo "progress push failed"
    rm -rf "$d"
  done
}

dice_watchdog() {  # 第 60 epoch 後若最近 40 個 epoch 的 pseudo Dice 全是 0 → 停止訓練，不再浪費 GPU 時間
  while sleep 600; do
    local f; f=$(ls -t "$FOLD_DIR"/training_log_*.txt 2>/dev/null | head -1); [ -n "$f" ] || continue
    local n; n=$(grep -c "Epoch time" "$f" || true)
    [ "${n:-0}" -ge 60 ] || continue
    if ! grep "Pseudo dice" "$f" | tail -40 | grep -qvE "float32\(0\.0\)"; then
      echo "WATCHDOG: pseudo Dice 0 for the last 40 epochs at epoch $n, stopping training"
      pkill -f nnUNetv2_train; return
    fi
  done
}

finish() {
  local rc=$?
  [ -n "${PROG_PID:-}" ] && kill "$PROG_PID" 2>/dev/null || true
  [ -n "${DOG_PID:-}" ] && kill "$DOG_PID" 2>/dev/null || true
  echo "exit code $rc; elapsed: $(( ($(date +%s) - t0) / 60 )) min. log: $LOG"
  upload_weights || true
  local d; d=$(mktemp -d); collect "$d"
  if push_dir "$d" "lidc results $STAMP on $(cat "$d/gpu.txt" 2>/dev/null) rc=$rc"; then
    echo "results pushed: https://github.com/$REPO/tree/pilot-results/$OUT_DIR"
  else
    echo "results NOT pushed"
  fi
}
trap finish EXIT

command -v unzip >/dev/null || (apt-get update -qq && apt-get install -y -qq unzip) || true
preflight
pipi -r requirements.txt
pipi "nnunetv2>=2.6,<2.9" "pylidc==0.2.3" "sqlalchemy<2" "pydicom<3" "setuptools<81" idc-index
# 2026-11-02 第一次全量訓練：映像檔內是 torch 2.8.0 nightly（dev 版）+ torch.compile，
# 第 17 epoch 起 pseudo Dice 一直是 0、loss 停在 -0.2 → 換正式版 torch、關掉 compile
if python -c "import torch,sys;sys.exit(0 if 'dev' in torch.__version__ else 1)"; then
  pipi "torch==2.7.1" "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cu128
fi
export nnUNet_compile=${nnUNet_compile:-f}
python -c "import torch;print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
df -h "$WS" | tail -1

if [ ! -f "$nnUNet_raw/$FULL/dataset.json" ]; then
  if [ "${SKIP_SMOKE:-0}" != "1" ]; then
    echo "== 0/4 smoke test: 5 scans end to end (download, masks, plan, 1 epoch) before the 125 GB download =="
    SM=Dataset502_LIDCSmoke
    python -m scripts.download_idc --collection lidc_idri --out "$WS/smoke_dicom" --limit 5
    python -W ignore -m scripts.lidc_to_nnunet --dicom_root "$WS/smoke_dicom" --out "$nnUNet_raw/$SM" --workers 2
    if [ "$MODE" = "roi" ]; then
      python -m scripts.lidc_crops --src "$nnUNet_raw/$SM" --out "$nnUNet_raw/${SM}ROI" --workers 2
      rm -rf "$nnUNet_raw/$SM"; mv "$nnUNet_raw/${SM}ROI" "$nnUNet_raw/$SM"
    fi
    nnUNetv2_plan_and_preprocess -d 502 -c 3d_fullres -np 2 --verify_dataset_integrity
    cp "$nnUNet_raw/$SM/splits_final.json" "$nnUNet_preprocessed/$SM/"
    nnUNetv2_train 502 3d_fullres 0 -tr nnUNetTrainer_1epoch
    python -m scripts.lesion_metrics --pred "$nnUNet_results/$SM/nnUNetTrainer_1epoch__nnUNetPlans__3d_fullres/fold_0/validation" \
      --gt "$nnUNet_raw/$SM/labelsTr" --out "$WS/runs/smoke_lesion_metrics.json"
    rm -rf "$WS/smoke_dicom" "$nnUNet_raw/$SM" "$nnUNet_preprocessed/$SM" "$nnUNet_results/$SM"
    echo "smoke test ok"
  fi

  echo "== 1/4 download LIDC-IDRI CT from IDC =="
  python -m scripts.download_idc --collection lidc_idri --out "$DICOM" ${LIMIT:+--limit $LIMIT}
  du -sh "$DICOM"

  echo "== 2/4 pylidc consensus masks -> nnU-Net dataset =="
  python -W ignore -m scripts.lidc_to_nnunet --dicom_root "$DICOM" --out "$nnUNet_raw/$FULL" --workers "$NP"
  rm -rf "$DICOM"   # 轉完就刪原始 DICOM（IDC 隨時可重新下載）
fi

if [ "$MODE" = "roi" ] && [ ! -f "$nnUNet_raw/$DS/dataset.json" ]; then
  echo "== 2b/4 nodule-centred ROIs =="
  python -m scripts.lidc_crops --src "$nnUNet_raw/$FULL" --out "$nnUNet_raw/$DS" --workers "$NP"
fi

echo "== 3/4 nnU-Net plan + preprocess (3d_fullres) =="
if [ ! -f "$nnUNet_preprocessed/$DS/nnUNetPlans.json" ] || [ ! -d "$nnUNet_preprocessed/$DS/nnUNetPlans_3d_fullres" ]; then
  nnUNetv2_plan_and_preprocess -d "$DSID" -c 3d_fullres -np "$NP" --verify_dataset_integrity
fi
cp "$nnUNet_raw/$DS/splits_final.json" "$nnUNet_preprocessed/$DS/splits_final.json"   # 病人層級切分
df -h "$WS" | tail -1

echo "== 4/4 train fold $FOLD ($TRAINER) =="
progress_loop & PROG_PID=$!
dice_watchdog & DOG_PID=$!
CONT=""; [ -f "$FOLD_DIR/checkpoint_latest.pth" ] && CONT="--c"
nnUNetv2_train "$DSID" 3d_fullres "$FOLD" -tr "$TRAINER" $CONT

echo "== lesion-level metrics on fold $FOLD validation =="
python -m scripts.lesion_metrics --pred "$FOLD_DIR/validation" --gt "$nnUNet_raw/$DS/labelsTr" \
  --out "$WS/runs/lesion_metrics.json" || echo "lesion metrics failed (voxel Dice is still in validation_summary.json)"
cat "$FOLD_DIR/validation/summary.json" | python3 -c "import json,sys;print('foreground mean:',json.load(sys.stdin)['foreground_mean'])" || true

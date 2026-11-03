"""
train_luna25.py — LUNA25 結節惡性度（良性 vs 惡性）3D 分類，病人層級 K-fold 交叉驗證

設計：
  - 輸入：官方 nodule blocks 重新取樣成 64³、視野 50 mm 的 patch
  - 模型：MONAI 3D ResNet-18（從頭訓練，不需下載預訓練權重，避免 pod 上網路失敗）
  - 每個 fold：測試 fold 只用來報告；早停用的是從訓練 fold 再切出的 15% 病人
  - 報告：各 fold AUC、合併 out-of-fold（OOF）AUC 與 bootstrap 95% CI、AP
  - 省錢：AMP、早停、每個 fold 做完就落盤，被收回的 spot pod 重跑會跳過已完成的 fold

用法（repo 根目錄）：
  python -m scripts.train_luna25 --csv data/luna25/LUNA25_Public_Training_Development_Data.csv \
      --blocks data/luna25/nodule_blocks --out runs/luna25
  煙霧測試：加 --limit 200 --epochs 1 --folds 2
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from monai.networks.nets import resnet18, resnet50
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from src.datasets.luna25 import filter_available, get_loaders, inner_split, load_annotations, make_folds

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


ARCHS = ("resnet18", "medicalnet18", "medicalnet50")


def build_model(arch: str = "resnet18", pretrained: bool = True):
    """resnet18 = 從頭訓練（原本的基準）；medicalnet18/50 = MedicalNet 預訓練骨幹（23 個醫療資料集，
    權重從 Hugging Face TencentMedicalNet 下載）+ 新的線性分類頭"""
    if arch == "resnet18":
        return resnet18(spatial_dims=3, n_input_channels=1, num_classes=1).to(DEVICE)
    fn, shortcut, bias, feat = {"medicalnet18": (resnet18, "A", True, 512),
                                "medicalnet50": (resnet50, "B", False, 2048)}[arch]
    backbone = fn(pretrained=pretrained, spatial_dims=3, n_input_channels=1, feed_forward=False,
                  shortcut_type=shortcut, bias_downsample=bias, conv1_t_stride=2)
    return nn.Sequential(backbone, nn.Dropout(0.2), nn.Linear(feat, 1)).to(DEVICE)


FLIPS = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]   # 8 種翻轉（z, y, x 軸）


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def bootstrap_auc(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    aucs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            aucs.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))] if aucs else [None, None]


def train_epoch(model, loader, opt, loss_fn, scaler, amp):
    model.train()
    tot, n = 0.0, 0
    for b in loader:
        x, y = b["image"].to(DEVICE, non_blocking=True), b["label"].to(DEVICE)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=amp):
            loss = loss_fn(model(x).squeeze(1).float(), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        tot += loss.item() * len(y)
        n += len(y)
    return tot / max(n, 1)


@torch.no_grad()
def predict(model, loader, amp, tta=False):
    """tta=True 時另外回傳 8 種翻轉平均的機率"""
    model.eval()
    ids, ys, ps, pt = [], [], [], []
    for b in loader:
        x = b["image"].to(DEVICE, non_blocking=True)
        probs = []
        for f in (FLIPS if tta else FLIPS[:1]):
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=amp):
                logit = model(torch.flip(x, f) if f else x).squeeze(1)
            probs.append(torch.sigmoid(logit.float()).cpu().numpy())
        ps.append(probs[0])
        pt.append(np.mean(probs, 0))
        ys.append(b["label"].numpy())
        ids += list(b["id"])
    if tta:
        return ids, np.concatenate(ys), np.concatenate(ps), np.concatenate(pt)
    return ids, np.concatenate(ys), np.concatenate(ps)


def run_fold(k, df, cfg, amp):
    fdir = os.path.join(cfg.out, f"fold{k}")
    done = os.path.join(fdir, "test_predictions.csv")
    if os.path.exists(done):
        print(f"fold {k}: already done, skipping")
        return pd.read_csv(done, dtype={"AnnotationID": str}), json.load(open(os.path.join(fdir, "metrics.json")))
    os.makedirs(fdir, exist_ok=True)
    test = df[df.fold == k].reset_index(drop=True)
    train, val = inner_split(df[df.fold != k].reset_index(drop=True), seed=cfg.seed + k)
    assert not (set(train.group) & set(val.group)) and not (set(train.group) & set(test.group))
    print(f"fold {k}: train {len(train)} (mal {int(train.label.sum())}) | val {len(val)} "
          f"(mal {int(val.label.sum())}) | test {len(test)} (mal {int(test.label.sum())})")

    tl, vl = get_loaders(train, val, cfg.blocks, cfg.size_px, cfg.size_mm, cfg.batch_size, cfg.num_workers)
    _, test_l = get_loaders(train.head(cfg.batch_size), test, cfg.blocks, cfg.size_px, cfg.size_mm,
                            cfg.batch_size, cfg.num_workers)
    torch.manual_seed(cfg.seed + k)
    model = build_model(cfg.arch)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler(DEVICE, enabled=amp)
    loss_fn = nn.BCEWithLogitsLoss()   # 類別不平衡已由加權抽樣處理

    best, stale, hist, t0 = -1.0, 0, [], time.time()
    best_path = os.path.join(fdir, "best.pt")
    for ep in range(cfg.epochs):
        loss = train_epoch(model, tl, opt, loss_fn, scaler, amp)
        _, vy, vp = predict(model, vl, amp)
        auc = safe_auc(vy, vp)
        sched.step()
        hist.append({"epoch": ep, "loss": round(loss, 5), "val_auc": round(auc, 5)})
        print(f"fold {k} epoch {ep:03d} | loss {loss:.4f} | val AUC {auc:.4f} | {time.time() - t0:.0f}s")
        if auc > best or best < 0:
            best, stale = auc, 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
            if cfg.patience and stale >= cfg.patience:
                print(f"fold {k}: early stop at epoch {ep}")
                break

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    ids, y, p, ptta = predict(model, test_l, amp, tta=True)
    pred = pd.DataFrame({"AnnotationID": ids, "label": y.astype(int), "prob": p, "prob_tta": ptta, "fold": k})
    m = {"fold": k, "test_auc": safe_auc(y, p), "test_auc_tta": safe_auc(y, ptta), "best_val_auc": best, "epochs_run": len(hist),
         "minutes": round((time.time() - t0) / 60, 1), "n_test": len(test), "n_test_malignant": int(y.sum()),
         "history": hist}
    json.dump(m, open(os.path.join(fdir, "metrics.json"), "w"), indent=2)
    pred.to_csv(done, index=False)
    print(f"fold {k}: test AUC {m['test_auc']:.4f}")
    return pred, m


def main(cfg):
    os.makedirs(cfg.out, exist_ok=True)
    amp = cfg.amp and DEVICE == "cuda"
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
    if not cfg.num_workers:   # pod 內 nproc 會回報整台主機核心數，改讀 cgroup 限制
        from src.preprocess.pipeline import _container_limits
        cfg.num_workers = max(1, min(_container_limits()[0], 8))
    df = filter_available(load_annotations(cfg.csv), cfg.blocks)
    if cfg.limit:   # 煙霧測試：分層取前 N 顆
        df = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(len(g), max(cfg.limit // 2, 4)), random_state=cfg.seed)).reset_index(drop=True)
    df = make_folds(df, cfg.folds, cfg.seed)
    df[["AnnotationID", "group", "label", "fold"]].to_csv(os.path.join(cfg.out, "folds.csv"), index=False)
    print(f"{len(df)} nodules ({int(df.label.sum())} malignant) from {df.group.nunique()} patients | "
          f"{cfg.folds} folds | device {DEVICE} amp {amp}")

    t0 = time.time()
    preds, per_fold = [], []
    folds = range(cfg.folds) if cfg.only_fold is None else [cfg.only_fold]
    for k in folds:
        p, m = run_fold(k, df, cfg, amp)
        preds.append(p)
        per_fold.append({key: m.get(key) for key in ("fold", "test_auc", "test_auc_tta", "best_val_auc",
                                                       "epochs_run", "minutes")})
    oof = pd.concat(preds, ignore_index=True)
    oof.to_csv(os.path.join(cfg.out, "oof_predictions.csv"), index=False)
    aucs = [f["test_auc"] for f in per_fold]
    summary = {
        "task": "LUNA25 nodule malignancy (benign vs malignant), patient-level CV",
        "model": {"resnet18": "MONAI 3D ResNet-18 from scratch",
                  "medicalnet18": "MedicalNet ResNet-18 (pretrained, 23 datasets) + linear head",
                  "medicalnet50": "MedicalNet ResNet-50 (pretrained, 23 datasets) + linear head"}[cfg.arch]
                 + f", {cfg.size_px}^3 patch, {cfg.size_mm:g} mm FOV, lr {cfg.lr:g}",
        "arch": cfg.arch,
        "n_nodules": int(len(oof)), "n_malignant": int(oof.label.sum()),
        "oof_auc": safe_auc(oof.label, oof.prob), "oof_auc_95ci": bootstrap_auc(oof.label, oof.prob),
        "oof_auc_tta": safe_auc(oof.label, oof.prob_tta) if "prob_tta" in oof else None,
        "oof_auc_tta_95ci": bootstrap_auc(oof.label, oof.prob_tta) if "prob_tta" in oof else None,
        "oof_average_precision": float(average_precision_score(oof.label, oof.prob)) if oof.label.nunique() == 2 else None,
        "fold_auc_mean": float(np.nanmean(aucs)), "fold_auc_std": float(np.nanstd(aucs)),
        "per_fold": per_fold, "wall_hours": round((time.time() - t0) / 3600, 3), "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "licence_note": "LUNA25 labels are CC BY-NC 4.0: research use only, not for commercial product training",
    }
    json.dump(summary, open(os.path.join(cfg.out, "summary.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_fold"}, indent=2))


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/luna25/LUNA25_Public_Training_Development_Data.csv")
    p.add_argument("--blocks", default="data/luna25/nodule_blocks")
    p.add_argument("--out", default="runs/luna25")
    p.add_argument("--arch", choices=ARCHS, default="resnet18")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--only_fold", type=int, default=None, help="只跑某一個 fold（例如先跑 fold 0 看趨勢）")
    p.add_argument("--size_px", type=int, default=64)
    p.add_argument("--size_mm", type=float, default=50.0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0, help="0 = 依容器實際 CPU 自動決定（上限 8）")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())

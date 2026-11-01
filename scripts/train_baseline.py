"""
train_baseline.py — LUNA16 假陽性削減 baseline（真結節 vs 候選假陽性）
對應研發執行手冊 1.2：3D DenseNet + 病人層級驗證

省錢設計（RunPod 計時計費）：
  - 預設 AMP 混合精度（GPU 上約 2 倍速、顯存減半）
  - 64³ patch、早停（--patience），不浪費 epoch
  - 每個 epoch 存 last.pt，可用便宜的可中斷（spot）pod，被收回後 --resume 續跑
  - 結束時印出實際 GPU 時數，方便對帳

用法（在 repo 根目錄）：
  python -m scripts.train_baseline --meta data/processed/meta.csv \
      --npy_dir data/processed/npy --candidates data/luna16/candidates_V2.csv
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from monai.networks.nets import DenseNet121
from sklearn.metrics import roc_auc_score
from torch import nn

from src.datasets.luna16 import get_loaders, load_meta, make_candidate_df, split_by_series

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model():
    """輕量 3D DenseNet，單通道輸入、二分類輸出"""
    return DenseNet121(
        spatial_dims=3, in_channels=1, out_channels=2,
        init_features=32, growth_rate=16, block_config=(4, 8, 12, 10),
    ).to(DEVICE)


def train_one_epoch(model, loader, opt, loss_fn, scaler, amp):
    model.train()
    tot, correct, losses = 0, 0, 0.0
    for batch in loader:
        x, y = batch["image"].to(DEVICE, non_blocking=True), batch["label"].to(DEVICE)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=amp):
            logits = model(x)
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        tot += len(y)
    return losses / max(tot, 1), correct / max(tot, 1)


@torch.no_grad()
def evaluate(model, loader, amp):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        x = batch["image"].to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=amp):
            logits = model(x)
        ps.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
        ys.append(batch["label"].numpy())
    y_true, y_prob = np.concatenate(ys), np.concatenate(ps)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    return auc, y_true, y_prob


def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.out, exist_ok=True)
    amp = cfg.amp and DEVICE == "cuda"
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    meta = load_meta(cfg.meta)
    df = make_candidate_df(cfg.candidates, meta, neg_per_pos=cfg.neg_per_pos, seed=cfg.seed)
    if cfg.limit:
        df = df.head(cfg.limit)
    train_df, val_df = split_by_series(df, cfg.val_subset, seed=cfg.seed)
    overlap = set(train_df.seriesuid) & set(val_df.seriesuid)
    assert not overlap, f"patient leakage: {len(overlap)} series in both splits"
    print(f"train {len(train_df)} patches / {train_df.seriesuid.nunique()} series "
          f"(pos {int(train_df.label.sum())}) | val {len(val_df)} / "
          f"{val_df.seriesuid.nunique()} series (pos {int(val_df.label.sum())}) | device {DEVICE} amp {amp}")

    patch = (cfg.patch_size,) * 3
    train_loader, val_loader = get_loaders(train_df, val_df, meta, cfg.npy_dir, patch,
                                           cfg.batch_size, cfg.num_workers)

    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler(DEVICE, enabled=amp)
    # 負例已降採樣至 1:3，預設不再加權（pos_weight=1），避免重複修正類別不平衡
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, cfg.pos_weight], device=DEVICE))

    start_epoch, best_auc, stale, history = 0, 0.0, 0, []
    last = os.path.join(cfg.out, "last.pt")
    if cfg.resume and os.path.exists(last):
        ck = torch.load(last, map_location=DEVICE)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_auc, stale, history = ck["epoch"] + 1, ck["best_auc"], ck["stale"], ck["history"]
        print(f"resumed from epoch {start_epoch}")

    t0 = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, loss_fn, scaler, amp)
        val_auc, _, _ = evaluate(model, val_loader, amp)
        sched.step()
        history.append({"epoch": epoch, "loss": tr_loss, "acc": tr_acc, "val_auc": val_auc})
        print(f"epoch {epoch:03d} | loss {tr_loss:.4f} | acc {tr_acc:.3f} | val AUC {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc, stale = val_auc, 0
            torch.save(model.state_dict(), os.path.join(cfg.out, "best.pt"))
        else:
            stale += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_auc": best_auc, "stale": stale,
                    "history": history}, last)
        if cfg.patience and stale >= cfg.patience:
            print(f"early stop: no val AUC gain for {cfg.patience} epochs")
            break

    hours = (time.time() - t0) / 3600
    summary = {"best_val_auc": best_auc, "epochs_run": len(history), "wall_hours": round(hours, 3),
               "note": "AUC on a 1:3 sampled patch set; not comparable to LUNA16 CPM"}
    with open(os.path.join(cfg.out, "metrics.json"), "w") as f:
        json.dump({**summary, "history": history}, f, indent=2)
    print(json.dumps(summary))


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--meta", default="data/processed/meta.csv")
    p.add_argument("--npy_dir", default="data/processed/npy")
    p.add_argument("--candidates", default="data/luna16/candidates_V2.csv")
    p.add_argument("--out", default="runs/baseline")
    p.add_argument("--val_subset", type=int, default=None,
                   help="多 subset 時使用官方 fold 作驗證；單 subset 試跑時自動依 series 切 20%%")
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5, help="早停；0 = 關閉")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--neg_per_pos", type=int, default=3)
    p.add_argument("--pos_weight", type=float, default=1.0)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="只取前 N 筆（煙霧測試用）")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())

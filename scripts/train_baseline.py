"""
train_baseline.py — LUNA16 baseline 模型訓練（假陽性削減 / 真假結節分類）
對應研發執行手冊 1.2：3D DenseNet-121 + FROC 驗證
相依：monai, torch, dataset.py
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121
from monai.metrics import ROCAUCMetric
from sklearn.metrics import roc_auc_score
from dataset import make_candidate_df, get_loaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model():
    """3D DenseNet-121，單通道輸入、二分類輸出"""
    return DenseNet121(
        spatial_dims=3, in_channels=1, out_channels=2,
        init_features=32, growth_rate=16, block_config=(4, 8, 12, 10),
    ).to(DEVICE)


def train_one_epoch(model, loader, opt, loss_fn):
    model.train()
    tot, correct, losses = 0, 0, 0.0
    for batch in loader:
        x, y = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        losses += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        tot += len(y)
    return losses / tot, correct / tot


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        x, y = batch["image"].to(DEVICE), batch["label"]
        prob = torch.softmax(model(x), 1)[:, 1].cpu().numpy()
        ps.append(prob)
        ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    return roc_auc_score(y_true, y_prob), y_true, y_prob


def main(cfg):
    torch.manual_seed(42)
    df = make_candidate_df(cfg.annotations, cfg.candidates)
    train_loader, val_loader = get_loaders(
        df, cfg.ct_root, batch_size=cfg.batch_size)

    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, cfg.pos_weight]).to(DEVICE))
    best_auc = 0.0

    for epoch in range(cfg.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, loss_fn)
        val_auc, _, _ = evaluate(model, val_loader)
        sched.step()
        print(f"epoch {epoch:03d} | loss {tr_loss:.4f} | "
              f"acc {tr_acc:.3f} | val AUC {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), f"{cfg.out}/best.pt")
    print(f"best val AUC = {best_auc:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", default="data/luna16/annotations.csv")
    p.add_argument("--candidates", default="data/luna16/candidates_V2.csv")
    p.add_argument("--ct_root", default="data/processed/npz")
    p.add_argument("--out", default="runs/baseline")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--pos_weight", type=float, default=3.0,
                   help="正樣本（真結節）類別權重，對應 1:3 負樣本比例")
    main(p.parse_args())

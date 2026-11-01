# RunPod 最小試跑手冊（LUNA16 subset0）

目標：用最少的錢證明整條管線可用 —— 下載 → 前處理 → 預訓練偵測器 FROC/CPM → 假陽性削減 baseline。
**預估花費：約 US$1–3**（以下時數為估計值，價格為 RunPod 2026-11-01 公告的 on-demand 價）。

## 1. 建立 pod（約 5 分鐘）

| 設定 | 建議值 | 理由 |
|---|---|---|
| GPU | **RTX A5000 24 GB（$0.27/h）**；缺貨時 RTX 3090（$0.50/h）或 RTX 4090（$0.89/h） | 試跑不需要 A100/H100；24 GB 足夠偵測器推論與 64³ patch 訓練 |
| Cloud | **Community Cloud** | LUNA16 是公開、已去識別的資料，可用較便宜的 Community Cloud。**醫院資料或任何非公開資料一律不可放 Community Cloud** |
| 計費 | 可用 Spot／Interruptible（若有提供） | 訓練每個 epoch 都存 `last.pt`，被收回後重開 pod 再跑一次腳本即自動續跑 |
| Template | RunPod PyTorch 2.4（或本 repo 的 GHCR 映像） | 已含 CUDA/torch，省安裝時間 |
| Container disk | 20 GB | |
| Volume disk（/workspace） | **40 GB** | subset0 解壓約 13 GB + 前處理約 3 GB + bundle/權重；zip 解壓後自動刪除 |

## 2. 在 pod 裡執行（Web Terminal 或 SSH）

```bash
cd /workspace
git clone https://github.com/tonythetiger168/pulmoguard-ai.git
cd pulmoguard-ai
bash scripts/runpod_pilot.sh
```

腳本跑完（成功或失敗）會**自動停止 pod**，GPU 停止計費；/workspace 的資料與結果保留。
想跑完保持開機：`KEEP_POD=1 bash scripts/runpod_pilot.sh`。

## 3. 看結果

- `/workspace/runs/monai_detector/froc.json` —— 預訓練偵測器在 subset0 的敏感度與 **CPM**（公開文獻約 0.85–0.90）。
  bundle 權重以官方 fold0（val = subset0）訓練，所以 subset0 對它是未見資料。
- `/workspace/runs/baseline/metrics.json` —— 假陽性削減分類器的驗證 AUC（依病人切分；1:3 抽樣，不等同 CPM）。
- `/workspace/runs/pilot_*.log` —— 完整紀錄與總耗時，可用來對帳。

## 4. 結束後省錢

- 不再需要時**刪除 pod**（Terminate）；只停止（Stop）仍會收 volume 磁碟費（$0.20/GB/月，閒置時）。
- 要保留結果：先把 `runs/` 下載到本機或推到雲端儲存，再刪 pod。

## 5. 估計時數與花費（RTX A5000，$0.27/h；皆為估計）

| 步驟 | 估計時間 |
|---|---|
| 安裝套件＋下載 subset0（6.8 GB） | 15–30 分 |
| 前處理 89 例 | 10–20 分 |
| MONAI 偵測器推論 89 例（AMP） | 30–60 分 |
| baseline 訓練（≤30 epoch，早停） | 15–45 分 |
| **合計** | **約 1.5–3 小時 ≈ US$0.4–0.8 GPU 費**，加磁碟與重跑緩衝約 **US$1–3** |

若 A5000 推論顯存不足（OOM），改開 RTX 4090／3090 重跑即可（已完成的步驟會自動跳過）。

## 6. 之後擴大時的省錢原則

1. **先跑一個 fold，不要一開始就跑 10 折**；10 折只在要報正式數字時跑。
2. 偵測器先用預訓練權重評估／微調，不從頭訓練。
3. 優先用 24 GB 的便宜卡（A5000／3090），只有顯存真的不夠才換 L40S／A100。
4. 前處理與下載用 CPU 或最便宜的 pod，不佔用貴的 GPU。
5. 一律開 AMP（混合精度）、早停、每 epoch 存檔；搭配 Spot pod。
6. 每個腳本結尾自動 `runpodctl stop pod`，避免忘記關機。

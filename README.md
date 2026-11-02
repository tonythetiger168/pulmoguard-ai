# PulmoGuard AI

**AI + LLM + Agent + LDCT 肺癌篩檢全流程智能平台**

結合 3D 影像 AI（CADe/CADx）、大型語言模型（LLM）與智慧代理（Agent），覆蓋低劑量電腦斷層（LDCT）肺癌篩檢的完整臨床路徑：掃描 → 結節偵測 → 惡性表徵 → 報告生成 → 追蹤管理 → 陽性轉介。

---

## 為什麼是 PulmoGuard？

| 全球現況 | PulmoGuard 的差異化 |
|---|---|
| FDA 已核准 eyonis LCS、RevealAI-Lung（僅偵測與表徵） | 結節追蹤管理自動化（Agent 排程，全球產品空白帶） |
| NLST 以歐美吸菸男性為主 | 台灣不吸菸女性肺腺癌族群適配 |
| 無台灣本土 AI+LDCT 成本效益證據 | 首份本土經濟學研究（ISPOR 格式） |

## 系統架構

```
L4 人機協作層   PACS 整合介面 / 醫師審核工作台
L3 Agent 層    追蹤管理 / 召回排程 / 報告品管 / 試驗招募
L2 LLM 層     本地部署醫療微調模型 + RAG（台灣肺癌指引）
L1 影像 AI 層  CADe 偵測 / CADx radiomics / nnU-Net 分割
L0 資料層     DICOM / PACS / 去識別化管線
```

## Repo 結構

```
pulmoguard-ai/
├── configs/           # 訓練組態（YAML）
├── src/
│   ├── preprocess/    # DICOM→NIfTI→patch 前處理管線
│   ├── datasets/      # MONAI Dataset（LUNA16/LIDC）
│   ├── models/        # 3D CNN / DenseNet / nnU-Net
│   ├── cadx/          # radiomics + 風險模型
│   └── agents/        # 追蹤管理 Agent（M12+）
├── scripts/           # download_tcia.py / train_baseline.py
├── data/              # raw/（DICOM，gitignored）processed/
├── docs/              # 計劃書 / 執行手冊 / 資料庫目錄
├── tests/  notebooks/  runs/
├── Dockerfile  requirements.txt
```

## Quickstart

```bash
# 1. 環境（二擇一）
pip install -r requirements.txt
# 或
docker build -t lung-ai . && docker run --gpus all -it -v $(pwd):/app lung-ai

# 2. 最小試跑資料：LUNA16 subset0（89 例，6.8 GB，CC BY 4.0）
bash scripts/download_luna16.sh data/luna16 0

# 3. 前處理 → 預訓練偵測器 FROC/CPM → 假陽性削減 baseline
python -m src.preprocess.pipeline --luna_root data/luna16 --out data/processed --subsets 0
python -m scripts.eval_monai_detector --luna_root data/luna16 --subset 0
python -m scripts.train_baseline --meta data/processed/meta.csv --npy_dir data/processed/npy

# 在 RunPod 一鍵跑完（跑完自動關機）：見 docs/RunPod_最小試跑手冊.md
bash scripts/runpod_pilot.sh

# 4. 正式驗證：先單一 fold，再 nnDetection 10-fold → CPM
```

CI 以合成資料驗證：重採樣幾何、世界座標→體素換算、病人層級切分、FROC/CPM 計算，以及前處理→訓練→續跑的端到端流程。合成資料上的數字不代表真實效能。

## 開發路線圖

| 階段 | 里程碑 |
|---|---|
| M0–2 | LUNA16 baseline（本 repo 目前位置） |
| M3–9 | CADe（nnDetection）＋分割（nnU-Net） |
| M8–14 | CADx radiomics 風險模型（LIDC 惡性度序數） |
| M10–16 | LLM 報告模組（本地模型 + RAG） |
| M12–18 | **Agent 追蹤管理模組**（零監管障礙，最快商轉） |
| M18+ | TFDA → FDA 510(k)（predicate: eyonis LCS K251474） |

## 文件

- [提案計劃書](docs/AI_LLM_Agent_LDCT_提案計劃書.md) — 策略層（30 個月、預算、團隊）
- [研發執行手冊](docs/研發執行手冊_資料管線與申請範本.md) — 戰術層（管線規格＋CDAS 範本）
- [RunPod 最小試跑手冊](docs/RunPod_最小試跑手冊.md) — 約 US$1–3 跑通整條管線
- [全球資料庫目錄](docs/global_lung_ai_databases.csv) — 17 個訓練資料庫

## 聲明

本專案處於研究開發階段，所有模型輸出僅供研究用途，不構成醫療診斷建議。臨床使用前須通過 IRB 審查與 TFDA/FDA 取證。

## License

程式碼：Apache-2.0（見 LICENSE）。模型權重與資料依各資料集授權另議。

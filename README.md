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

# 2. 抓 pilot 資料（50 例，約 6 GB）
python scripts/download_tcia.py --collection LIDC-IDRI --limit 50

# 3. 前處理 + 訓練 baseline
python -m src.preprocess.pipeline
python scripts/train_baseline.py --epochs 60

# 4. 正式驗證：nnDetection 10-fold → Grand Challenge CPM
```

沙盒已驗證：前處理邏輯（HU 窗/重採樣/球形偽標籤）與 3D CNN 訓練迴路（合成資料 test AUC 0.956）。

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
- [全球資料庫目錄](docs/global_lung_ai_databases.csv) — 17 個訓練資料庫

## 聲明

本專案處於研究開發階段，所有模型輸出僅供研究用途，不構成醫療診斷建議。臨床使用前須通過 IRB 審查與 TFDA/FDA 取證。

## License

TBD（建議：程式 MIT／Apache-2.0，模型權重與資料另議）

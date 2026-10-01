# Dockerfile — AI+LDCT 訓練環境（GPU 版）
# 建置：docker build -t lung-ai .
# 執行：docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/runs:/app/runs lung-ai
#      python download_tcia.py --limit 50

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

# 系統工具（dcm2niix 官方 apt 無套件，從 GitHub release 安裝二進位）
RUN apt-get update && apt-get install -y --no-install-recommends \
      wget unzip git && \
    wget -q https://github.com/rordenlab/dcm2niix/releases/download/v1.0.20240202/dcm2niix_lnx.zip && \
    unzip -o dcm2niix_lnx.zip -d /usr/local/bin/ && rm dcm2niix_lnx.zip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 相依
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 專案程式
COPY . .

# 預設：印環境資訊（勿預設跑訓練，方便互動除錯）
CMD ["python", "-c", "import torch,monai,SimpleITK;print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'| monai',monai.__version__,'| SimpleITK',SimpleITK.Version_VersionString())"]

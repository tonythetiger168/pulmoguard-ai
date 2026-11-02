# Dockerfile — AI+LDCT 訓練環境（GPU 版，亦作為 RunPod pod 映像）
# 建置：docker build -t lung-ai .
# 執行：docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/runs:/app/runs lung-ai \
#        python -m scripts.train_baseline
# 注意：.dockerignore 排除 data/ runs/ 等，資料一律以 volume 掛載，不進映像

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

# 系統工具（dcm2niix 官方 apt 無套件，從 GitHub release 安裝二進位）
RUN apt-get update && apt-get install -y --no-install-recommends \
      wget unzip git && \
    wget -q https://github.com/rordenlab/dcm2niix/releases/download/v1.0.20240202/dcm2niix_lnx.zip && \
    unzip -o dcm2niix_lnx.zip -d /usr/local/bin/ && rm dcm2niix_lnx.zip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 相依（torch 已在 base image 內）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 專案程式
COPY . .

# 非 root 執行
RUN useradd -m -u 1000 app && chown -R app /app
USER app

CMD ["python", "-c", "import torch,monai,SimpleITK;print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'| monai',monai.__version__,'| SimpleITK',SimpleITK.Version_VersionString())"]

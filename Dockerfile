# 使用 x86 架構的 Python 3.8 slim 映像
FROM --platform=linux/amd64 python:3.8-slim

# 安裝必要套件（build tools + glx 為 baseline .so 執行依賴）
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && apt-get clean

# 設定工作目錄
WORKDIR /app

# 複製所有專案檔案到容器中
COPY . .

# 安裝 Python 套件
RUN pip install --no-cache-dir -r requirement.txt

# 預設執行指令（你也可以改）
# 這裡改成 repo 內實際存在的入口檔
CMD ["python", "start_game_plain.py"]

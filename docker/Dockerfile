# panotrack —— 360° ERP 全景视频实时单目标跟踪（赛事提交镜像）
#
# 断网自包含原则：构建期一次性安装全部依赖并 COPY 源码，运行期零网络访问。
# 构建命令（赛事要求 linux/amd64，在仓库根目录执行）：
#   docker build --platform linux/amd64 -f docker/Dockerfile -t panotrack:latest .
# 运行示例（挂载数据目录）：
#   docker run --rm --platform linux/amd64 \
#     -v /path/to/data:/data panotrack:latest \
#     --frames /data/frames --init /data/init.txt --out /data/results.txt
# 注：不使用 # syntax 远程指令（避免构建期拉取 docker.eiffel 前端）；
# 基础镜像用本地已缓存的 python:3.12-slim-bookworm（等价 python:3.12-slim）。
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖（利用构建缓存），再 COPY 源码
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY panotrack ./panotrack
COPY configs ./configs
# LightFC/VitTrack ONNX 模型打进镜像(断网自包含;评测机零网络访问)
COPY models ./models
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 正式入口：文件协议 CLI（参数见 panotrack/cli.py）
ENTRYPOINT ["python", "-m", "panotrack.cli"]

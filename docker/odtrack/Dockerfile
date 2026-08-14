# GRT-360 ODTrack 精度版提交镜像（Arena 平台官方协议：BFoV 输出）
#
# 断网自包含原则：构建期一次性安装全部依赖并 COPY 上游源码与权重，
# 运行期零网络访问。上游代码与权重不进 Git，由构建脚本从
# artifacts/server_snapshot/ 组装临时构建上下文（见 docker/odtrack/build_odtrack_image.sh）。
#
# 评测协议（Arena 平台，2026-08-14 核实）：
#   输入 /mnt/dataset（只读）：<seq>/video.mp4 + <seq>/init.txt（BFoV: clon,clat,fov_h,fov_v）
#   输出 /mnt/result（读写）：<seq>.txt，每行 BFoV，行号=帧号；丢失帧 0,0,0,0
#   容器无参自启动跑完全部序列，退出码 0（非 0 判失败）。
# 正式运行需 GPU：docker run --rm --gpus all \
#   -v <测试集>:/mnt/dataset:ro -v <输出>:/mnt/result <镜像>
# 环境变量 DATASET_DIR / RESULT_DIR 可覆盖默认挂载路径。
ARG BASE_IMAGE=pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="GRT-360 ODTrack ERP 精度版提交镜像 (Arena BFoV)"
LABEL org.opencontainers.image.source="https://github.com/A3435331841/instan"
LABEL grt360.method="ODTrack Base + ERP 三平铺适配 + BFoV 官方协议"
LABEL grt360.metrics="AUC=0.5792135073 SR=0.6531941586 FPS=8.9945 (120 sequences, 2x RTX 3090)"
LABEL grt360.checkpoint.sha256="2fba6ddeb826014ac0bb871623406d16c3a162afbf09accb49312b526c21068e"
LABEL grt360.upstream="https://github.com/GXNU-ZhongLab/ODTrack"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/odtrack \
    DATASET_DIR=/mnt/dataset \
    RESULT_DIR=/mnt/result

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libglib2.0-0 libgl1 libjpeg-turbo8 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/grt360/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/grt360/requirements.txt

# ODTrack 上游源码（lib/ 与 experiments/ 配置）与官方权重（构建脚本去重，
# 快照 output/ 内不再保留副本，镜像中仅此一份）
COPY odtrack_src/ /opt/odtrack/
COPY ODTrack_ep0300.pth.tar /opt/models/ODTrack_ep0300.pth.tar
COPY integrations/ /opt/grt360/integrations/
COPY scripts/odtrack_360vot.py /opt/grt360/scripts/odtrack_360vot.py

RUN python -c "import torch, torchvision, cv2, numpy, easydict, yacs, timm, einops, pandas, PIL, scipy; print('torch', torch.__version__, 'numpy', numpy.__version__)"

WORKDIR /opt/odtrack
# Arena 平台官方协议入口：无参自启动，读 /mnt/dataset 写 /mnt/result（BFoV）
ENTRYPOINT ["python", "/opt/grt360/integrations/odtrack/arena_protocol.py"]

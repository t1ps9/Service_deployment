FROM python:3.10

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    supervisor \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV DOCKER_IP=localhost

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth /root/.cache/torch/hub/checkpoints/
COPY . /app

EXPOSE 8080
EXPOSE 9090

CMD ["supervisord", "-c", "/app/supervisord.conf"]
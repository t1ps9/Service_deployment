FROM python:3.10

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth /root/.cache/torch/hub/checkpoints/
COPY . /app

EXPOSE 8080
EXPOSE 9090

CMD ["supervisord", "-c", "/app/supervisord.conf"]
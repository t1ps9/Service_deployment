FROM python:3.10


WORKDIR /app
# ENV DOCKER_IP=localhost

COPY requirements.txt /app/requirements.txt
COPY grpc_my.py /app
COPY task1_app.py /app

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth /app
COPY . /app
RUN python -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. ./proto/inference.proto

EXPOSE 8080 9090
RUN chmod +x /app/start.sh


CMD ["sh", "start.sh"]

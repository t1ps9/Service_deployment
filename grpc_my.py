import logging
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
import requests
from PIL import Image
from io import BytesIO
from torchvision import transforms

import inference_pb2
import inference_pb2_grpc

logging.basicConfig(level=logging.INFO)


def initialize_model():
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
    model.eval()
    return model, weights.meta['categories']


model, coco_categories = initialize_model()


def fetch_image(url: str) -> Image.Image:
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return img
    except Exception as e:
        logging.error("Изображение не загрузилось", e)
        raise ValueError("Неверный url изображения")


def prepare_image(url: str) -> torch.Tensor:
    img = fetch_image(url)
    tensor = transforms.ToTensor()(img)
    return tensor


def infer(tensor: torch.Tensor, threshold: float = 0.75) -> list:
    with torch.no_grad():
        output = model([tensor])[0]
    results = []
    for lbl, score in zip(output["labels"], output["scores"]):
        if score.item() >= threshold:
            results.append(coco_categories[lbl])
    return results


class InstanceDetectorServicer(inference_pb2_grpc.InstanceDetectorServicer):
    def Predict(self, request, context):
        url = request.url
        logging.info("gRPC запрос: %s", url)
        try:
            image_tensor = prepare_image(url)
        except ValueError as err:
            context.set_details(str(err))
            context.set_code(grpc_server.StatusCode.INVALID_ARGUMENT)
            return inference_pb2.InstanceDetectorOutput(objects=[])

        detected = infer(image_tensor)
        return inference_pb2.InstanceDetectorOutput(objects=detected)


def serve():
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InstanceDetectorServicer_to_server(InstanceDetectorServicer(), server)
    listen_addr = "[::]:9090"
    server.add_insecure_port(listen_addr)
    server.start()
    logging.info("gRPC сервер запущен и слушает %s", listen_addr)
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("Останавливаем gRPC сервер")
        server.stop(0)


if __name__ == '__main__':
    serve()

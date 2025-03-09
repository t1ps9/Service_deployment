import logging
from flask import Flask, request, jsonify, Response
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from torchvision import transforms
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
model.eval()

coco_categories = weights.meta['categories']
http_inference_counter = Counter('app_http_inference_count', 'Multiprocess metric')


def download_image(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except Exception as e:
        logging.error("Изображение не загрузилось", e)
        raise ValueError("Неверный url изображения")


def preprocess_image(image: Image.Image):
    tensor = transforms.ToTensor()(image)
    return tensor


def run_model(image_tensor: torch.Tensor):
    with torch.no_grad():
        return model([image_tensor])[0]


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    if not data or 'url' not in data:
        return jsonify({'error': 'В JSON должен присутствовать ключ "url"'}), 400

    image_url = data['url']
    try:
        img = download_image(image_url)
    except ValueError as err:
        return jsonify({'error': str(err)}), 400

    img_tensor = preprocess_image(img)
    predictions = run_model(img_tensor)

    threshold = 0.75
    detected_objects = []
    for label, score in zip(predictions['labels'], predictions['scores']):
        if score.item() >= threshold:
            detected_objects.append(coco_categories[label])

    http_inference_counter.inc()

    return jsonify({'objects': detected_objects})


@app.route('/metrics', methods=['GET'])
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

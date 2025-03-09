import logging
from flask import Flask, request, jsonify, Response
import grpc
import inference_pb2
import inference_pb2_grpc
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

chan = grpc.insecure_channel('localhost:9090')
prot = inference_pb2_grpc.InstanceDetectorStub(chan)

http_inference_counter = Counter('app_http_inference_count', 'Multiprocess metric')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    if not data or 'url' not in data:
        return jsonify({'error': "Нет url"}), 400

    image_url = data['url']
    try:
        chan = grpc.insecure_channel('localhost:9090')
        prot = inference_pb2_grpc.InstanceDetectorStub(chan)
        grpc_request = inference_pb2.InstanceDetectorInput(url=image_url)
        grpc_response = prot.Predict(grpc_request)
        detected_objects = list(grpc_response.objects)
    except grpc.RpcError as e:
        return jsonify({'error': str(e.details())}), 500

    http_inference_counter.inc()
    return jsonify({'objects': detected_objects})


@app.route('/metrics', methods=['GET'])
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

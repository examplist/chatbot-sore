from PIL import Image
import requests
import torch
import numpy
from flask import Flask, request, jsonify
import json
import cv2
from util import imgPred

dirRoot = "/home/jdh/Desktop/chatbot"

##@brief 학습된 모델을 로드하는 부분
path = dirRoot + "/ref/"
model_bedsore = torch.hub.load(
    dirRoot + "/y5", "custom", path=path + "best_bedsore.pt", source="local"
)
model_burn = torch.hub.load(
    dirRoot + "/y5", "custom", path=path + "best_burn.pt", source="local"
)

app = Flask(__name__)


@app.route("/", methods=["POST"])
def main():
    request_data = json.loads(request.get_data())
    link_pre = request_data["action"]["detailParams"]["sore_image"]["origin"]
    link = link_pre.replace("List(", "").replace(")", "")
    print("link", link)
    imgOriginal = Image.open(requests.get(link, stream=True).raw)
    imgResized = imgOriginal.resize([300, 300])
    decode_image = numpy.array(imgResized)

    # box : 바운딩 박스 좌표
    # cx : 바운딩 박스 중심 x 좌표
    # cy : 바운딩 박스 중심 y 좌표
    label, confidence = imgPred(decode_image, model_bedsore, size=224)
    confidence = "%.1f" % (confidence * 100)
    print("snd_confidence = ", confidence)
    print("label", label)

    response = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": f"진단 결과: {label}, 예측 확률: {confidence}",
                    },
                },
            ],
        },
    }

    return jsonify(response)


app.run("0.0.0.0", debug=True, port=5000)

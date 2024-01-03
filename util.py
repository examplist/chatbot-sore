"""
@brief 이미지에서 객체를 검출해주는 함수 파일
@author Jeaseung Kim
@date 2023-01-10
@version 2.0.0
"""

import numpy as np
import cv2
import time


def imgPred(image, model, size=224):
    # img predict
    image_np = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    imgs = [
        image_np,
    ]

    # start = time.perf_counter_ns()

    # 객체 검출 - 이미지를 input 사이즈와 함께 입력하여 객체 검출
    results = model(imgs, size)

    # duration = (time.perf_counter_ns() - start)
    # print(f"검출 추론 과정 : {duration // 1000000}ms.")

    ps = results.pandas().xyxy[0]

    if len(ps.values.tolist()) == 0:
        label = "Others"
        confidence = 0
        return label, confidence

    p_list = ps.values.tolist()[0]

    # print('xmin : ', p_list[0])
    # print('ymin : ', p_list[1])
    # print('xmax : ', p_list[2])
    # print('ymax : ', p_list[3])
    # print('confidence : ', p_list[4])
    # print('class : ', p_list[5])
    # print('name : ', p_list[6])

    label = p_list[6]
    confidence = p_list[4]

    others = ["unclassified pressure ulcer", "suspected deep tissue injury"]

    if label in others:
        label = "Others"
        return label, confidence

    return label, confidence

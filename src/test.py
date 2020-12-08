import tensorflow as tf
import numpy as np
from model import InpaitingModel
from dataset import Dataset
import matplotlib.pyplot as plt
import cv2
import os
import sys
import json
from canny import canny

# folder = sys.argv[1]
job_folder = "job_test"
with open(os.path.join(job_folder, 'configurations.json'), 'r') as f:
    config = json.load(f)
model_config = config["model_config"]
model_config["model_ckpoint_dir"] = os.path.join(job_folder, model_config["model_ckpoint_dir"])

model = InpaitingModel(model_config)

il = os.listdir("datasets/sample/image")
ml = os.listdir("datasets/sample/mask")

for ii, (i, m) in enumerate(zip(il, ml)):
    i = cv2.imread(os.path.join("datasets/sample/image", i))
    m = cv2.imread(os.path.join("datasets/sample/mask", m))
    mask = cv2.resize(cv2.cvtColor(m, cv2.COLOR_BGR2GRAY), (i.shape[1], i.shape[1]))
    i = cv2.resize(i, (i.shape[1], i.shape[1]))
    mask = np.uint8(mask>0)[None, :,:,None]
    gray_image = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)
    edge = canny(gray_image, guassian_blur_sigma=1.0, sobel_size=3, lowThresholdRatio=0.25, highThresholdRatio=0.3)[None, :,:,None]
    gray_image = gray_image[None, :,:,None]
    new_edge = model.infer_edge(gray_image, edge, mask)
    cv2.imwrite("datasets/sample/edge_result/{}.png".format(ii))


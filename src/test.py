import tensorflow as tf
import numpy as np
from model import InpaitingModel
from dataset import Dataset
import cv2
import os
import sys
import json
from canny import canny

job_folder = sys.argv[1]
# job_folder = "job_test"
with open(os.path.join(job_folder, 'configurations.json'), 'r') as f:
    config = json.load(f)
model_config = config["model_config"]
model_config["use_pretrained_weights"] = True
model_config["model_ckpoint_dir"] = os.path.join(job_folder, model_config["model_ckpoint_dir"])
dataset_config = config["dataset_config"]

model = InpaitingModel(model_config)
dataset = Dataset(dataset_config, dataset_config["img_test_flist"], dataset_config["mask_test_flist"]).get_color_dataset().take(30)

for i, (edge, clr_img, mask) in enumerate(dataset):
    img_out_dir = os.path.join(model_config['model_ckpoint_dir'], "{}/".format(i))
    if not os.path.exists(img_out_dir):
        os.mkdir(img_out_dir)
    cv2.imwrite(os.path.join(img_out_dir, "true_edge.png"), edge[:,:,0].numpy() * 255)
    cv2.imwrite(os.path.join(img_out_dir, "mask.png"), mask[:,:,0].numpy() * 255)
    cv2.imwrite(os.path.join(img_out_dir, "original_color.png"), clr_img.numpy()[:,:,(2,1,0)])
    gray_img = tf.image.rgb_to_grayscale(clr_img)
    predicted_edge = model.infer_final_edge(tf.expand_dims(gray_img, 0), tf.expand_dims(edge, 0), tf.expand_dims(mask, 0))
    predicted_edge_img = tf.squeeze(predicted_edge).numpy()
    cv2.imwrite(os.path.join(img_out_dir, "predicted_edge.png"), predicted_edge_img)
    ground_edge_inpainting = model.infer_final_inpainting(tf.expand_dims(clr_img, 0), tf.expand_dims(edge, 0), tf.expand_dims(mask, 0))
    ground_edge_inpainting = tf.squeeze(ground_edge_inpainting).numpy()
    cv2.imwrite(os.path.join(img_out_dir, "groud_edge_inpainting.png"), ground_edge_inpainting[:,:,(2,1,0)])
    predicted_edge_inpainting = model.infer_final_inpainting(tf.expand_dims(clr_img, 0), predicted_edge / 255, tf.expand_dims(mask, 0))
    predicted_edge_inpainting = tf.squeeze(predicted_edge_inpainting).numpy()
    cv2.imwrite(os.path.join(img_out_dir, "predicted_edge_inpainting.png"), predicted_edge_inpainting[:,:,(2,1,0)])






# for ii, (i, m) in enumerate(zip(il, ml)):
#     i = cv2.imread(os.path.join("datasets/sample/image", i))
#     m = cv2.imread(os.path.join("datasets/sample/mask", m))
#     mask = cv2.resize(cv2.cvtColor(m, cv2.COLOR_BGR2GRAY), (i.shape[1], i.shape[1]))
#     i = cv2.resize(i, (i.shape[1], i.shape[1]))
#     mask = np.uint8(mask>0)[None, :,:,None]
#     gray_image = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)
#     edge = canny(gray_image, guassian_blur_sigma=1.0, sobel_size=3, lowThresholdRatio=0.25, highThresholdRatio=0.3)[None, :,:,None]
#     gray_image = gray_image[None, :,:,None]
#     new_edge = model.infer_edge(gray_image, edge, mask)
#     cv2.imwrite("datasets/sample/edge_result/{}.png".format(ii))


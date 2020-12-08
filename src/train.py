import tensorflow as tf
import numpy as np
from model import InpaitingModel
from dataset import Dataset
import os
import sys
import json

job_folder = sys.argv[1]
# job_folder = "job1"
with open(os.path.join(job_folder, 'configurations.json'), 'r') as f:
    config = json.load(f)
dataset_config = config["dataset_config"]
model_config = config["model_config"]
model_config["model_ckpoint_dir"] = os.path.join(job_folder, model_config["model_ckpoint_dir"])
dataset = Dataset(dataset_config, dataset_config["img_train_flist"], dataset_config["mask_train_flist"], "train")
model = InpaitingModel(model_config)

batch_size = dataset_config["batch_size"]

if model_config["edge"]["train"]:
    edge_dataset = dataset.get_edge_dataset().shuffle(batch_size * 50).batch(batch_size).prefetch(10)
    model.train_edge_part(edge_dataset, model_config["edge"]["train_epoch"], model_config["edge"]["ckpoint_step"], (dataset.size + batch_size - 1)//batch_size)

if model_config["clr"]["train"]:
    color_dataset = dataset.get_color_dataset().shuffle(batch_size * 50).batch(batch_size).prefetch(10)
    model.train_inpainting_part(color_dataset, model_config["clr"]["train_epoch"], model_config["clr"]["ckpoint_step"], (dataset.size + batch_size - 1)//batch_size)

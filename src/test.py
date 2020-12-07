import tensorflow as tf
import numpy as np
from model import InpaitingModel
from dataset import Dataset
import os
import sys
import json

# folder = sys.argv[1]
job_folder = "job1"
with open(os.path.join(job_folder, 'configurations.json'), 'r') as f:
    config = json.load(f)
dataset_config = config["dataset_config"]
model_config = config["model_config"]
model_config["model_ckpoint_dir"] = os.path.join(job_folder, model_config["model_ckpoint_dir"])
dataset = Dataset(dataset_config, dataset_config["img_train_flist"], dataset_config["mask_train_flist"], "train")
model = InpaitingModel(model_config)
model.load_checkpoint('eg')
model.load_checkpoint('ig')

if model_config["edge"]["train"]:
    edge_dataset = dataset.get_edge_dataset().shuffle(dataset_config["batch_size"] * 50).batch(dataset_config["batch_size"]).prefetch(2)
    model.train_edge_part(edge_dataset, model_config["edge"]["train_epoch"])

if model_config["clr"]["train"]:
    edge_dataset = dataset.get_edge_dataset().shuffle(dataset_config["batch_size"] * 50).batch(dataset_config["batch_size"]).prefetch(2)
    model.train_inpainting_part(edge_dataset, model_config["clr"]["train_epoch"])
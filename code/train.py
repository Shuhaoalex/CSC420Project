import tensorflow as tf
import numpy as np
from model import *
from losses import *
from dataset import Dataset
import os

model_config = {
    "edge": {
        "lamb_adv": 1,
        "lamb_fm": 10,
        "generator" : [
            {"mode":"conv", "chnl": 16, "ksize":(5,5), "name":"conv1"},
            {"mode":"conv", "chnl": 32, "stride":(2,2), "name":"conv2_downsample"},
            {"mode":"conv", "chnl": 32, "name":"conv3"},
            {"mode":"conv", "chnl": 64, "stride":(2,2), "name":"conv4_downsample"},
            {"mode":"conv", "chnl": 64, "name":"conv5"},
            {"mode":"conv", "chnl": 64, "d_factor":(2,2), "name":"conv6_astrous"},
            {"mode":"conv", "chnl": 64, "d_factor":(4,4), "name":"conv7_astrous"},
            {"mode":"conv", "chnl": 64, "name":"conv10"},
            {"mode":"deconv", "chnl": 32, "name":"conv11_upsample"},
            {"mode":"conv", "chnl": 32, "name":"conv12"},
            {"mode":"deconv", "chnl": 16, "name":"conv13_upsample"},
            {"mode":"conv", "chnl": 1, "name":"conv14"},
        ]
    },
    "clr": {
        "lamb_l1": 1,
        "lamb_adv": 0.1,
        "lamb_perc": 0.1,
        "lamb_style": 250, # paper use 250 here
        "generator": [
            {"mode":"conv", "chnl": 32, "ksize":(5,5), "name":"conv1"},
            {"mode":"conv", "chnl": 64, "stride":(2,2), "name":"conv2_downsample"},
            {"mode":"conv", "chnl": 64, "name":"conv3"},
            {"mode":"conv", "chnl": 128, "stride":(2,2), "name":"conv4_downsample"},
            {"mode":"conv", "chnl": 128, "name":"conv5"},
            {"mode":"conv", "chnl": 128, "d_factor":(2,2), "name":"conv6_astrous"},
            {"mode":"conv", "chnl": 128, "d_factor":(4,4), "name":"conv7_astrous"},
            {"mode":"conv", "chnl": 128, "d_factor":(8,8), "name":"conv8_astrous"},
            {"mode":"conv", "chnl": 128, "d_factor":(16,16), "name":"conv9_astrous"},
            {"mode":"conv", "chnl": 128, "name":"conv10"},
            {"mode":"deconv", "chnl":64, "name":"conv11_upsample"},
            {"mode":"conv", "chnl": 64, "name":"conv12"},
            {"mode":"deconv", "chnl": 32, "name":"conv13_upsample"},
            {"mode":"conv", "chnl": 3, "name":"conv14"},
        ]
    },
    "model_ckpoint_dir": ".",
    "use_pretrained_weights": False
}


dataset_config = {"img_train_flist":"../datasets/celeba_train.flist",
    "img_test_flist":"../datasets/celeba_test.flist",
    "img_validation_flist":"../datasets/celeba_validation.flist",
    "mask_train_flist":"../datasets/mask_train.flist",
    "mask_validation_flist":"../datasets/mask_validation.flist",
    "mask_test_flist":"../datasets/mask_test_test.flist",
    "sigma":2, "input_size":256
}

dataset = Dataset(dataset_config, dataset_config["img_train_flist"], dataset_config["mask_train_flist"], training=True)
dataset = dataset.dataset.shuffle(100).batch(10).prefetch(200)

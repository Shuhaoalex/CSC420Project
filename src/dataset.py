import os
import glob
import random
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from skimage.feature import canny
from PIL import Image

class Dataset(object):
    def __init__(self, config, img_list, mask_list, mode):
        super(Dataset, self).__init__()

        # image list and mask list
        self.img_list = self.load_flist(img_list)
        self.mask_list = self.load_flist(mask_list)

        # type of dataset: either train, test, or validation
        self.mode = mode
        
        # read parameter values from the config file
        self.input_size = config["input_size"]
        self.sigma = config["sigma"]
        self.size = len(self.img_list)

        # process the data we use
        self.edge_dataset = None
        self.color_dataset = None
        self.generate_edge_dataset()
        self.generate_color_dataset()

    def get_edge_dataset(self):
        return self.edge_dataset

    def get_color_dataset(self):
        return self.color_dataset
    
    def get_size(self):
        return self.size

    # TODO: Add reference
    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist
        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist
            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]
        return []
    
    def preprocess_img_mask_edge_data(self, path, mode=None):
        size = self.input_size
        path = path.numpy().decode("utf-8")
        image = cv2.imread(path)
        color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r,c = color_image.shape[0], color_image.shape[1]

        # 1. ground truth part
        # 1.1 ground truth color image
        # for training and validation data, first check to crop the original color image
        if self.mode != "test":
            if r != c:
                color_image = self.crop_square_image(color_image)
        # resize image to the size specified in config file
        color_image = cv2.resize(color_image, (size,size))
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        color_image = color_image.astype(np.float32)

        # 1.2 input mask is 0(black) for missing foreground region and 255(white) for background
        # random select a mask for training
        if self.mode != "test":
            mask_idx = random.randint(0, len(self.mask_list) - 1)
        else:
            mask_idx = np.where(self.img_list == path)[0][0]

        mask = cv2.imread(self.mask_list[mask_idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if self.mode != "test":
            if mask.shape[0] != mask.shape[1]:
                mask = self.crop_square_image(mask)
        mask = cv2.resize(mask, (size, size))
        # make sure values are either 255 or 0 after resizing with interpolation 
        bool_mask = mask > 0
        # final output for mask: 0(black) for missing foreground region and 1(white) for background
        mask = bool_mask.astype(np.float32)
        
        # 1.3 ground truth edge without any masked regions
        sigma = self.sigma
        gray_blur_image = cv2.GaussianBlur(gray_image, (5, 5), sigma)
        gray_image = gray_image.astype(np.float32)
        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions
        edge_mask = None if self.mode != "test" else bool_mask
        # canny
        edge_map = canny(gray_blur_image, sigma=sigma, mask=edge_mask).astype(np.float32) # background is 0, foreground is 1
        
        # finalize output
        # normalize gray and color image from [0,255] to [-1,1]
        color_image = 2*(color_image - np.amin(color_image)) / (np.amax(color_image) - np.amin(color_image)) - 1
        gray_image =  2*(gray_image - np.amin(gray_image)) / (np.amax(gray_image) - np.amin(gray_image)) - 1
        # expand dimension to 3 channels
        gray_image = tf.expand_dims(gray_image, axis=2)
        mask = tf.expand_dims(mask, axis=2) # background is 1, foreground mask region is 0
        edge_map = tf.expand_dims(edge_map, axis=2) # background is 0, foreground is 1

        if mode == "edge":
            # TODO: check multiply first or normalize first
            masked_gray = mask * gray_image
            return masked_gray, edge_map, mask
        else:
            return edge_map, color_image, mask

    def preprocess_edge_dataset(self, path):
        return self.preprocess_img_mask_edge_data(path, mode="edge")

    def preprocess_color_dataset(self, path):
        return self.preprocess_img_mask_edge_data(path, mode="color")

    def generate_edge_dataset(self):
        self.edge_dataset = tf.data.Dataset.from_tensor_slices(self.img_list).map(lambda x:tf.py_function(self.preprocess_edge_dataset, inp=[x],Tout=[np.float32, np.float32, np.float32]))

    def generate_color_dataset(self):
        self.color_dataset = tf.data.Dataset.from_tensor_slices(self.img_list).map(lambda x:tf.py_function(self.preprocess_color_dataset, inp=[x],Tout=[np.float32, np.float32, np.float32]))

    # crop square image around the center
    def crop_square_image(self, img):
        r, c = img.shape[0], img.shape[1]
        center_r, center_c = r // 2, c//2
        side = min(r, c)
        left_idx, top_idx = center_c-side//2, center_r-side//2
        crop_img = img[top_idx:top_idx+side, left_idx:left_idx+side]

        return crop_img

def construct_dataset(config, mode=None):
    img_list, mask_list = None,None
    if mode == "train":
        img_list = config["img_train_flist"]
        mask_list = config["mask_train_flist"]
    elif mode == "validation":
        img_list = config["img_validation_flist"]
        mask_list = config["mask_validation_flist"]
    elif mode == "test":
        img_list = config["img_test_flist"]
        mask_list = config["mask_test_flist"]

    if img_list and mask_list:
        dataset = Dataset(config, img_list, mask_list, mode)
        return dataset

if __name__ == "__main__":
    # TODO: should initialize all variables to a config file
    config = {"img_train_flist":"../datasets/celeba_train.flist", \
        "img_test_flist":"../datasets/celeba_test.flist", \
            "img_validation_flist":"../datasets/celeba_validation.flist", \
                "mask_train_flist":"../datasets/mask_train.flist", \
                    "mask_validation_flist":"../datasets/mask_validation.flist", \
                        "mask_test_flist":"../datasets/mask_test_test.flist", \
                            "sigma":2, "input_size":256}
    
    # get neural network datasets
    # inputs: config file, dataset mode(train, test, validation)
    dataset = construct_dataset(config, mode="train")
    edge_dataset = dataset.get_color_dataset()
    color_dataset = dataset.get_edge_dataset()
    size = dataset.get_size()
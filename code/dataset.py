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
    def __init__(self, config, img_list, mask_list, training=False, validation=False, test=False):
        super(Dataset, self).__init__()

        # image list and mask list
        self.img_list = self.load_flist(img_list)
        self.mask_list = self.load_flist(mask_list)

        # type of dataset
        self.training = training
        self.validation = validation
        self.test = test
        
        # read parameter values from the config file
        self.input_size = config["input_size"]
        self.batch_size = config["batch_size"]
        self.sigma = config["sigma"]

        # process the data we use
        self.data_process()


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

    def preprocess_image_mask_edge(self,path):
        size = self.input_size
        path = path.numpy().decode("utf-8")
        image = cv2.imread(path)
        color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r,c = color_image.shape[0], color_image.shape[1]

        # 1. ground truth part
        # 1.1 ground truth color image
        # for training and validation data, first check to crop the original color image
        if not self.test:
            if r != c:
                color_image = self.crop_square_image(color_image)
        # resize image to the size specified in config file
        color_image = cv2.resize(color_image, (size,size))

        # 1.2 input mask is 0 for missing region and 255 for background
        # random select a mask for training
        if not self.test:
            mask_idx = random.randint(0, len(self.mask_list) - 1)
        else:
            mask_idx = np.where(self.img_list == path)[0][0]

        mask = cv2.imread(self.mask_list[mask_idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if not self.test:
            if mask.shape[0] != mask.shape[1]:
                mask = self.crop_square_image(mask)
        mask = cv2.resize(mask, (size, size))
        inv_mask = (mask > 0).astype(np.uint8) * 255 # make sure values are either 255 or 0 after resizing with interpolation # 255 for background, 0 for missing region
        mask = cv2.bitwise_not(inv_mask) # 0 for background, 255 for missing foreground region

        # 1.3 ground truth edge without any masked regions
        sigma = self.sigma
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        gray_blur_image = cv2.GaussianBlur(gray_image, (5, 5), sigma)
        if self.training or self.validation:
            edge_mask = None # set None to find ground truth edge map
        else:
            # in test mode images are masked (with masked regions),
            # using 'mask' parameter prevents canny to detect edges for the masked regions
            edge_mask = (1 - mask / 255).astype(np.bool)

        # canny
        inv_edge_map = canny(gray_blur_image, sigma=sigma, mask=edge_mask).astype(np.uint8) * 255 # background is 0, foreground is 255
        edge_map = cv2.bitwise_not(inv_edge_map)

        # TODO: check if all below inputs are needed
        # 2. inputs for network(masked grayscale image and masked edge map)
        if self.test:
            mask_gray_image = gray_image
            mask_edge_map = edge_map
        else:
            mask_gray_image_bg = cv2.bitwise_and(gray_image, gray_image, mask=inv_mask)
            mask_gray_image = cv2.bitwise_or(mask_gray_image_bg, mask)

            mask_edge_map = cv2.bitwise_or(edge_map, mask)

        # mask_color_image_bg = cv2.bitwise_and(color_image, color_image, mask=inv_mask)
        # mask_color_image = cv2.bitwise_or(mask_color_image_bg, np.stack([mask,mask,mask], axis=2))
        # cv2.imwrite("/Users/serenawang/csc420/project/datasets/hi.png", cv2.cvtColor(mask_color_image, cv2.COLOR_RGB2BGR))

        # convert all outputs to 3d
        gray_image = tf.expand_dims(gray_image, axis=2)
        mask = tf.expand_dims(mask, axis=2)
        edge_map = tf.expand_dims(edge_map, axis=2)
        mask_gray_image = tf.expand_dims(mask_gray_image, axis=2)
        mask_edge_map = tf.expand_dims(mask_edge_map, axis=2)

        return color_image, gray_image, mask, edge_map, mask_gray_image, mask_edge_map

    def data_process(self):
        # process original input image
        dataset = tf.data.Dataset.from_tensor_slices(self.img_list).map(lambda x:tf.py_function(self.preprocess_image_mask_edge, inp=[x],Tout=[np.uint8,np.uint8, np.uint8, np.uint8, np.uint8, np.uint8]))
        self.dataset = dataset.batch(self.batch_size)

    # crop square image around the center
    def crop_square_image(self, img):
        r, c = img.shape[0], img.shape[1]
        center_r, center_c = r // 2, c//2
        side = min(r, c)
        left_idx, top_idx = center_c-side//2, center_r-side//2
        crop_img = img[top_idx:top_idx+side, left_idx:left_idx+side]

        return crop_img
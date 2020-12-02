import os
import glob
import random
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from skimage.feature import canny

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        r,c = image.shape[0], image.shape[1]

        # for training and validation data, first check to crop
        if not self.test:
            if r != c:
                image = self.crop_square_image(image)
        
        # resize image to the size specified in config file
        image = cv2.resize(image, (size,size))

        # TODO: maybe return the opposite color of mask
        # random select a mask
        if not self.test:
            mask_idx = random.randint(0, len(self.mask_list) - 1)
        else:
            # in test mode, mask is not random
            mask_idx = self.img_list.index(path)
        
        mask = cv2.imread(self.mask_list[mask_idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if not self.test:
            if mask.shape[0] != mask.shape[1]:
                mask = self.crop_square_image(mask)
        mask = cv2.resize(mask, (size, size))
        mask = (mask > 0).astype(np.uint8) * 255
        result_mask = mask

        # TODO: check edge generation
        # generate edge
        sigma = self.sigma
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions
        mask = None if self.training else (1 - mask / 255).astype(np.bool)

        # canny
        edge = canny(gray_image, sigma=sigma, mask=mask).astype(np.float32)


        # TODO: Add other tensor objects if needed
        return image, result_mask, edge

    def data_process(self):
        # process original input image
        dataset = tf.data.Dataset.from_tensor_slices(self.img_list).map(lambda x:tf.py_function(self.preprocess_image_mask_edge, inp=[x],Tout=[np.float32, np.uint8, np.float32]))
        self.dataset = dataset.batch(self.batch_size)

    # crop square image around the center
    def crop_square_image(self, img):
        r, c = img.shape[0], img.shape[1]
        center_r, center_c = r // 2, c//2
        side = min(r, c)
        left_idx, top_idx = center_c-side//2, center_r-side//2
        crop_img = img[top_idx:top_idx+side, left_idx:left_idx+side]

        return crop_img

if __name__ == "__main__":
    # TODO: should initialize all variables to a config file
    config = {"img_train_flist":"../datasets/celeba_train_test.flist", \
        "img_test_flist":"../datasets/celeba_test.flist", \
            "img_validation_flist":"../datasets/celeba_validation.flist", \
                "mask_train_flist":"../datasets/mask_train_test.flist", \
                    "mask_test_flist":"../datasets/mask_validation.flist", \
                        "mask_test_flist":"../datasets/mask_test.flist", \
                            "batch_size":8, "sigma":2, "input_size":256}

    # initialize training dataset, training_dataset.dataset is of type tf.data.Dataset
    training_dataset = Dataset(config, config["img_train_flist"], config["mask_train_flist"], training=True, validation=False, test=False)

    # initialize validation dataset
    validation_dataset = Dataset(config, config["img_validation_flist"], config["mask_train_flist"], training=False, validation=True, test=False)

    # initialize test dataset
    test_dataset = Dataset(config, config["img_test_flist"], config["mask_test_flist"], training=False, validation=False, test=True)
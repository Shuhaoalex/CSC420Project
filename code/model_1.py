# input a non-connected edge and return a connected edge map
# masked grayscale image
# edge map
# image mask, 1 for missing region, 0 for background
# dimension?

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# filter_number should be greater than 3
def gateConvolution(concat_input, filter_number, k_size, stride, d_rate):
    # concat input should be a 4-D, do reshape on gray image
    output = tf.keras.layers.Conv2D(filter_number, k_size,
                                    strides=(stride, stride), padding="SAME", dilation_rate=d_rate,
                                    activation=None)(concat_input)
    # split the filters into two subparts along dimension 3
    gating, feature = tf.split(output, 2, 3)
    activate_feature = tf.nn.elu(feature)
    smooth_gating = tf.nn.sigmoid(gating)
    return activate_feature * smooth_gating

def gateDeconvolution(concat_input, height, width, filter_number):
    concat_input = tf.image.resize(concat_input,[height, width], method='nearest')
    return gateConvolution(concat_input, filter_number, 3, 1, 1)

class EdgeGenerator(tf.keras.Model):
    def __init__(self, img_shape):
        # img shape is width * height * channel
        super(EdgeGenerator, self).__init__()
        image_mask, edge_mask, gray_image = \
            layers.Input(img_shape),layers.Input(img_shape),layers.Input(img_shape)
        # Input
        concat_input = tf.concat((image_mask, edge_mask, gray_image), axis=3)

        # ----------- CONVOLUTION ------------- (need to change parameter)
        # number of filters in for loop
        intialize_number = 2
        # if running multiple convolution
        # filter_list = [intialize_number]
        # filter_list.extend([intialize_number * 2] * 2)
        # filter_list.extend([intialize_number * 4] * 9)
        # for i in range(len(filter_list)):
        concat_input = gateConvolution(concat_input, intialize_number, 3, 1, 1)

        # ----------- DECONVOLUTION -------------
        concat_input= gateDeconvolution(concat_input, 266, 266, intialize_number)
        # hyperbolic tangent
        edgeCompleteImage = tf.nn.tanh(concat_input)

        # ---------- SIZE_CHECK -----------
        print("finalEdgeSize")
        print(edgeCompleteImage.shape)

        self.edgeCompleteImage = tf.keras.Model(inputs=(image_mask, edge_mask, gray_image),
                                                outputs=edgeCompleteImage, name="edgeCompleteImage")

    def edgeComplete(self, image_mask, edge_mask, gray_image):
        return self.edgeCompleteImage(image_mask, edge_mask, gray_image)


class ImageGenerator(tf.keras.Model):
    def __init__(self, img_shape):
        # img_shape -->  width * height * channel
        super(ImageGenerator, self).__init__()
        # here the predict_edge_mask is size batch * height * width * channel
        image_mask, predict_edge_mask, incomplete_edge_mask, incomplete_color_image = \
            layers.Input(img_shape), layers.Input(img_shape),layers.Input(img_shape), layers.Input(img_shape)

        # ----------- Input -------------
        origin_input = tf.concat((incomplete_color_image, incomplete_edge_mask), axis=3)
        concat_input = image_mask * predict_edge_mask + (1 - image_mask) * origin_input[:, :, :, 0:3]
        # reshape we may not need it
        concat_input.set_shape(origin_input[:, :, :, 0:3].get_shape().as_list())

        # ----------- CONVOLUTION ------------- (need to change parameter)
        # number of filters in for loop
        intialize_number = 2
        # if running multiple convolution
        # filter_list = [intialize_number]
        # filter_list.extend([intialize_number * 2] * 2)
        # filter_list.extend([intialize_number * 4] * 9)
        # for i in range(len(filter_list)):
        concat_input = gateConvolution(concat_input, intialize_number, 3, 1, 1)

        # ----------- DECONVOLUTION -------------
        concat_input= gateDeconvolution(concat_input, 256, 256, intialize_number)
        # hyperbolic tangent
        ImageComplete = tf.nn.tanh(concat_input)
        # could not compute output tensor

         # ---------- SIZE_CHECK -----------
        print("finalImageSize")
        print(ImageComplete.shape)

        self.ImageComplete = tf.keras.Model(inputs=(image_mask, predict_edge_mask, incomplete_edge_mask,
                                            incomplete_color_image), outputs=ImageComplete,
                                            name="edgeCompleteImage")

    def ImageComplete(self, image_mask, predict_edge_mask, incomplete_edge_mask, incomplete_color_image):
        return self.ImageComplete(image_mask, predict_edge_mask, incomplete_edge_mask, incomplete_color_image)

#if __name__ == '__main__':
    # edge = np.ones((256,256,3))
    # a = EdgeGenerator((256,256,3))
    # b = a.edgeCompleteImage(edge,edge,edge)

    #image =ImageGenerator((256, 256, 3))
    #c = image.ImageComplete(edge, edge, edge, edge)

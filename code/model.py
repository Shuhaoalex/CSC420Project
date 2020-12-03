# input a non-connected edge and return a connected edge map
# masked grayscale image
# edge map
# image mask, 1 for missing region, 0 for background


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# filter_number should be greater than 3
def gateConvolution(concat_input, filter_number, k_size, stride, d_rate):
    # concat input should be a 4-D, do reshape on gray image
    output = tf.keras.layers.Conv2D(filter_number, k_size,
                                    stride=(stride, stride), padding="SAME", dilaiton_rate=d_rate,
                                    activation=None)(concat_input)
    # split the filters into two subparts along dimension 3
    gating, feature = tf.split(output, 2, 3)
    activate_feature = tf.nn.elu(feature)
    smooth_gating = tf.nn.sigmoid(gating)
    return activate_feature * smooth_gating

def gateDeconvolution(concat_input, height, width, filter_number):
    concat_input = tf.keras.backend.resize_images(concat_input, height, width,
                                                  data_format="channels_last",interpolation='nearest')
    return gateConvolution(concat_input, filter_number, 3, 1, 1)

class EdgeGenerator(tf.keras.Model):
    def __init__(self, image_mask, edge_mask, gray_image):
        super(EdgeGenerator, self).__init__()
        ## width * height
        input_image_mask, input_edge_mask, input_gray_image = \
            layers.Input(image_mask), layers.Input(edge_mask),layers.Input(gray_image)
        shape = input_image_mask.get_shape().as_list()
        ori_width = shape[0]
        ori_height = shape[1]
        image_mask_3d = input_image_mask.reshape(shape[0], shape[1], 1)
        edge_mask_3d = input_edge_mask.reshape(shape[0], shape[1], 1)
        gray_image_3d = input_gray_image.reshape(shape[0], shape[1], 1)
        # the initial input (batch_size, height, width, channel) default: chanel_last
        concat_input = tf.concat((image_mask_3d, edge_mask_3d, gray_image_3d), axis=0)

        # the initial input
        # layers input will change it to batch size????
        concat_input = tf.concat((image_mask, edge_mask, gray_image), axis=2)

        # number of filters in for loop
        intialize_number = 48
        filter_list = [intialize_number]
        filter_list.extend([intialize_number * 2] * 2)
        filter_list.extend([intialize_number * 4] * 9)
        for i in range(len(filter_list)):
                concat_input = gateConvolution(concat_input, filter_list[i], 3, 1, 1)
        concat_input = gateDeconvolution(concat_input, ori_width, ori_height, 2 * intialize_number)
        # hyperbolic tangent
        edgeCompleteImage = tf.nn.tanh(concat_input)
        self.edgeCompleteImage = tf.keras.Model(inputs=(input_image_mask, input_edge_mask, input_gray_image),
                                                outputs=edgeCompleteImage, name="edgeCompleteImage")

    def call(self, image_mask, edge_mask, gray_image):
        return self.edgeCompleteImage(image_mask, edge_mask, gray_image)


# class ImageGenerator(tf.keras.Model):
#     def __init__(self, image_mask, edge_mask, color_image):
#         super(ImageGenerator, self).__init__()
#         input_image_mask, input_edge_mask, input_gray_image = \
#             layers.Input(image_mask), layers.Input(edge_mask),layers.Input(color_image)
#         shape = input_image_mask.get_shape().as_list()
#         image_mask_3d = input_image_mask.reshape(shape[0], shape[1], 1)
#         edge_mask_3d = input_edge_mask.reshape(shape[0], shape[1], 1)
#         gray_image_3d = input_gray_image.reshape(shape[0], shape[1], 1)
#         edge_predict = EdgeGenerator(image_mask, edge_mask, color_image)
#         edge_predict = edge_predict.edgeCompleteImage(image_mask, edge_mask, color_image)
#
#
#         # the initial input (batch_size, height, width, channel) default: chanel_last
#
#         concat_input = tf.concat((image_mask_3d, edge_mask_3d, gray_image_3d), axis=0)
#
#         # number of filters in for loop
#         # make sure that the filter_number is even
#         filter_list = []
#         type_list = []
#         for i in range(len(filter_list)):
#             if type_list[i] == 'c':
#                 concat_input = gateConvolution(concat_input, filter_list[i], 3, 1)
#             elif type_list[i] == 'd':
#                 concat_input = gateDeconvolution(concat_input, filter_list[i])
#             else:
#                 # resize
#
#
#         colorImage = result
#         self.colorImage = tf.keras.Model(inputs=(input_image_mask, input_edge_mask, input_gray_image),
#                                                 outputs=colorImage, name="colorImage")
#
#     def colorImage(self, image_mask, edge_mask, gray_image):
#         return self.colorImage(image_mask, edge_mask, gray_image)

if __name__ == '__main__':
    e = EdgeGenerator(image_mask, edge_mask, gray_image)
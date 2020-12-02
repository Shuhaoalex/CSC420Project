import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers
import cv2 as cv

class EdgeDiscriminator(keras.Model):
    def __init__(self, img_shape):
        super(EdgeDiscriminator, self).__init__()
        input_x = layers.Input(img_shape)
        input_y = layers.Input(img_shape)
        curr_x = input_x
        curr_y = input_y
        FMLoss = 0
        conv_sz = [64, 128, 256, 512]
        for i, sz in enumerate(conv_sz):
            curr_layer = keras.Sequential((layers.Conv2D(filters=sz, kernel_size=[4,4], strides=[2,2], padding='same', use_bias=True, name="conv{}".format(i)), layers.LeakyReLU(0.2)))
            curr_x = curr_layer(curr_x)
            curr_y = curr_layer(curr_y)
            FMLoss += tf.reduce_mean(tf.abs(curr_x - curr_y))
        FMLoss /= 4

        flatten_layer = layers.Flatten()
        curr_x = flatten_layer(curr_x)
        dense_layer = layers.Dense(units=1, use_bias=True, name='fc')
        result_x = dense_layer(curr_x)

        self.LFM = keras.Model(inputs=(input_x, input_y), outputs=FMLoss, name="Lfm")
        self.pred_logit = keras.Model(inputs=input_x, outputs=result_x, name='pred_logit')
    
    def predict(self, img):
        return tf.nn.sigmoid(self.pred_logit(img))
    
    def feature_matching_loss(self, imga, imgb):
        return self.LFM(imga, imgb)
    
    def generator_loss(self, img):
        tmp = self.pred_logit(img)
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones(tf.shape(tmp)), logits=tmp))
    
    def discriminator_loss(self, fake_img, true_img):
        fake_logit = self.pred_logit(fake_img)
        true_logit = self.pred_logit(true_img)
        return tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        tf.ones(tf.shape(true_logit)), true_logit)) +\
               tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        tf.zeros(tf.shape(fake_logit)), fake_logit))


class InpaintingDescriminator(keras.Model):
    def __init__(self, img_shape):
        super(InpaintingDescriminator, self).__init__()
        self.model = keras.Sequential((
            layers.Input(img_shape),
            layers.Conv2D(filters=64, kernel_size=[4,4], strides=[2,2], padding="same", use_bias=True, name="conv0"),
            layers.LeakyReLU(0.2),
            layers.Conv2D(filters=128, kernel_size=[4,4], strides=[2,2], padding="same", use_bias=True, name="conv1"),
            layers.LeakyReLU(0.2),
            layers.Conv2D(filters=256, kernel_size=[4,4], strides=[2,2], padding="same", use_bias=True, name="conv2"),
            layers.LeakyReLU(0.2),
            layers.Conv2D(filters=512, kernel_size=[4,4], strides=[2,2], padding="same", use_bias=True, name="conv3"),
            layers.LeakyReLU(0.2),
            layers.Flatten(),
            layers.Dense(layers.Dense(units=1, use_bias=True, name='fc'))
        ))
    
    def call(self, img):
        return tf.nn.sigmoid(self.model(img))
    
    def generator_loss(self, img):
        tmp = self.model(img)
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones(tf.shape(tmp)), logits=tmp))
    
    def discriminator_loss(self, fake_img, true_img):
        fake_logit = self.model(fake_img)
        true_logit = self.model(true_img)
        return tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        tf.ones(tf.shape(true_logit)), true_logit)) +\
               tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        tf.zeros(tf.shape(fake_logit)), fake_logit))


class PerceptuaAndStylelLoss(keras.Model):
    def __init__(self):
        super(PerceptuaAndStylelLoss, self).__init__()
        vgg = keras.applications.VGG19(include_top=True, weights='imagenet')
        perceptual_out = list(map(lambda lname: vgg.get_layer(lname).output, ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]))
        style_out = list(map(lambda lname: vgg.get_layer(lname).output, ["block2_conv2", "block3_conv4", "block4_conv4", "block5_conv2"]))
        self.vgg_hijack = keras.Model(inputs=vgg.input, outputs=[perceptual_out, style_out])
    
    def compute_gram(self, x):
        shape = tf.shape(x)
        x = tf.reshape(x, (shape[0], shape[1] * shape[2], shape[3]))
        x_T = tf.transpose(x, perm=(0,2,1))
        return tf.matmul(x_T, x) / (shape[1] * shape[2] * shape[3])

    def call(self, a, b):
        a_pf, a_sf = self.vgg_hijack(a)
        b_pf, b_sf = self.vgg_hijack(b)

        PL = 0
        for af, bf in zip(a_pf, b_pf):
            PL += tf.reduce_mean(tf.abs(af - bf))
        PL /= 5

        SL = 0
        for af, bf in zip(a_sf, b_sf):
            cova = self.compute_gram(af)
            covb = self.compute_gram(bf)
            SL += tf.reduce_mean(tf.abs(cova - covb))
        SL /= 4
        return PL, SL
    



# a = EdgeLoss([224, 224, 3], [[64, 64], [128, 128]], [[0, 1], [0, 1], [0, 0, 1]], [1024, 1024, 1])
# img = cv.resize(cv.imread("Ladv.png"), (224, 224))[None, :, :, :].astype(np.float32)
# result = a(img, img, 1, 10)
# print(result)
# keras.utils.plot_model(a.LFM, "LFM.png")
# keras.utils.plot_model(a.Ladv, "Ladv.png")
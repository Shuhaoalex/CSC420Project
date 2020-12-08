import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers

class EdgeDiscriminator(keras.Model):
    def __init__(self, **kwargs):
        super(EdgeDiscriminator, self).__init__(**kwargs)
        conv_sz = [16, 32, 64, 64]
        self.convs = [
            keras.Sequential((
                layers.Conv2D(
                    filters=sz,
                    kernel_size=[4,4],
                    strides=[2,2],
                    padding='same',
                    use_bias=True, 
                ),
                layers.LeakyReLU(0.2)
            ), name="conv{}".format(i))
            for i, sz in enumerate(conv_sz)
        ]

        self.post_process = keras.Sequential((
            layers.Flatten(),
            layers.Dense(units=1, use_bias=True, name='fc')
        ), name="flatten")
    
    def pred_logit(self, img):
        for conv in self.convs:
            img = conv(img)
        return self.post_process(img)

    def feature_matching_loss(self, imga, imgb):
        loss = 0
        for conv in self.convs:
            imga = conv(imga)
            imgb = conv(imgb)
            loss += tf.reduce_mean(tf.abs(imga - imgb))
        return loss / 4.0

    def call(self, img):
        return tf.nn.sigmoid(self.pred_logit(img))
    
    def generator_loss(self, fake_img, true_img, lamb_adv=1.0, lamb_fm=10.0):
        fake_logit = self.pred_logit(fake_img)
        true_logit = self.pred_logit(true_img)
        adv_loss1 = tf.reduce_mean(
                           tf.nn.sigmoid_cross_entropy_with_logits(
                           tf.zeros(tf.shape(true_logit)), true_logit))
        adv_loss2 = tf.reduce_mean(
                           tf.nn.sigmoid_cross_entropy_with_logits(
                           tf.ones(tf.shape(fake_logit)), fake_logit))
        FMLoss = self.feature_matching_loss(fake_img, true_img)
        result = FMLoss * lamb_fm + (adv_loss1 + adv_loss2) * lamb_adv
        return result

    def discriminator_loss(self, fake_img, true_img):
        fake_logit = self.pred_logit(fake_img)
        true_logit = self.pred_logit(true_img)
        return tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        tf.ones(tf.shape(true_logit)), true_logit)) +\
               tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        tf.zeros(tf.shape(fake_logit)), fake_logit))


class InpaintingDiscriminator(keras.Model):
    def __init__(self, **kwargs):
        super(InpaintingDiscriminator, self).__init__(**kwargs)
        self.model = keras.Sequential((
            layers.Conv2D(filters=32, kernel_size=[4,4], strides=[2,2], padding="same", use_bias=True, name="conv0"),
            layers.LeakyReLU(0.2),
            layers.Conv2D(filters=64, kernel_size=[4,4], strides=[2,2], padding="same", use_bias=True, name="conv1"),
            layers.LeakyReLU(0.2),
            layers.Conv2D(filters=128, kernel_size=[4,4], strides=[2,2], padding="same", use_bias=True, name="conv2"),
            layers.LeakyReLU(0.2),
            layers.Conv2D(filters=128, kernel_size=[4,4], strides=[2,2], padding="same", use_bias=True, name="conv3"),
            layers.LeakyReLU(0.2),
            layers.Flatten(),
            layers.Dense(units=1, use_bias=True, name='fc')
        ))
    
    def call(self, img):
        return tf.nn.sigmoid(self.model(img))
    
    def generator_loss(self, fake_img, true_img):
        fake_logit = self.model(fake_img)
        true_logit = self.model(true_img)
        return tf.reduce_mean(
                           tf.nn.sigmoid_cross_entropy_with_logits(
                           tf.zeros(tf.shape(true_logit)), true_logit)) +\
                tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                        tf.ones(tf.shape(fake_logit)), fake_logit))

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
    def __init__(self, **kwargs):
        super(PerceptuaAndStylelLoss, self).__init__(trainable=False, **kwargs)
        vgg = keras.applications.VGG19(include_top=True, weights='imagenet')
        perceptual_out = list(map(lambda lname: vgg.get_layer(lname).output, ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]))
        style_out = list(map(lambda lname: vgg.get_layer(lname).output, ["block2_conv2", "block3_conv4", "block4_conv4", "block5_conv2"]))
        self.vgg_hijack = keras.Model(inputs=vgg.input, outputs=[perceptual_out, style_out])
    
    def compute_gram(self, x):
        shape = tf.shape(x)
        x = tf.reshape(x, (shape[0], shape[1] * shape[2], shape[3]))
        x_T = tf.transpose(x, perm=(0,2,1))
        return tf.matmul(x_T, x) / tf.cast(tf.reduce_prod(shape[1:]), tf.float32)

    def call(self, fake_img, real_img, lamb_p, lamb_s):
        means = -tf.constant([103.939, 116.779, 123.68])
        pre_processed_fake = tf.nn.bias_add((fake_img + 1) * 127.5, means)
        pre_processed_real = tf.nn.bias_add((real_img + 1) * 127.5, means)
        a_pf, a_sf = self.vgg_hijack(pre_processed_fake)
        b_pf, b_sf = self.vgg_hijack(pre_processed_real)

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
        return PL * lamb_p + SL * lamb_s


def reconstruction_loss(fake_img, true_img):
    return tf.reduce_mean(tf.abs(fake_img - true_img))

if __name__ == "__main__":
    a = EdgeDiscriminator()
    print(a)
    # a = EdgeLoss([224, 224, 3], [[64, 64], [128, 128]], [[0, 1], [0, 1], [0, 0, 1]], [1024, 1024, 1])
    # img = cv.resize(cv.imread("Ladv.png"), (224, 224))[None, :, :, :].astype(np.float32)
    # result = a(img, img, 1, 10)
    # print(result)
    # keras.utils.plot_model(a.LFM, "LFM.png")
    # keras.utils.plot_model(a.Ladv, "Ladv.png")

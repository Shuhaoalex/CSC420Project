import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class GatedConv2d(layers.Layer):
    def __init__(self, filter_number, ksize=(3,3), strides=(1,1), dilation_rate=(1,1), **kwargs):
        super(GatedConv2d, self).__init__(**kwargs)
        self.conv = layers.Conv2D(
            filter_number * 2,
            kernel_size=ksize,
            strides=strides,
            padding="same",
            dilation_rate=dilation_rate,
            use_bias=True,
            activation=None)

    def call(self, prev):
        convolved = self.conv(prev)
        gating, feature = tf.split(convolved, 2, 3)
        activate_feature = tf.nn.elu(feature)
        smooth_gating = tf.nn.sigmoid(gating)
        return activate_feature * smooth_gating


class GatedDeconv2d(layers.Layer):
    def __init__(self, filter_number, ksize=(3,3), strides=(2,2), dilation_rate=(1,1), **kwargs):
        super(GatedDeconv2d, self).__init__(**kwargs)
        self.conv = layers.Conv2DTranspose(
            filter_number * 2,
            kernel_size=ksize,
            strides=strides,
            padding="same",
            dilation_rate=dilation_rate,
            use_bias=True,
            activation=None)
    
    def call(self, prev):
        convolved = self.conv(prev)
        gating, feature = tf.split(convolved, 2, 3)
        activate_feature = tf.nn.elu(feature)
        smooth_gating = tf.nn.sigmoid(gating)
        return activate_feature * smooth_gating


class GatedConvGenerator(keras.Model):
    def __init__(self, config, **kwargs):
        """
        config is a list of dictionary containing parameter for each layer
        {
            "mode": "conv" or "deconv",
            "chnl": int,
            "ksize": (int, int),          --optional, default (3,3)
            "stride": (int, int),         --optional, default (1,1) for conv, (2,2) for deconv
            "d_factor": (int, int)        --optional, default (1,1)
            "name": string                --optional, default conv_i
        }
        """
        super(GatedConvGenerator, self).__init__(**kwargs)
        self.convs = keras.Sequential()
        for i, c in enumerate(config):
            if c["mode"] == "gconv":
                self.convs.add(
                    GatedConv2d(
                        c["chnl"],
                        ksize=c.get("ksize", (3,3)),
                        strides=c.get("stride", (1,1)),
                        dilation_rate=c.get("d_factor", (1,1)),
                        name=c.get("name", "conv{}".format(i))
                    )
                )
            elif c["mode"] == "gdeconv":
                self.convs.add(
                    GatedDeconv2d(
                        c["chnl"],
                        ksize=c.get("ksize", (3,3)),
                        strides=c.get("stride", (2,2)),
                        dilation_rate=c.get("d_factor", (1,1)),
                        name=c.get("name", "conv{}".format(i))
                    )
                )
            elif c["mode"] == "conv":
                self.convs.add(layers.Conv2D(
                    c["chnl"],
                    kernel_size=c.get("ksize", (3,3)),
                    strides=c.get("stride", (1,1)),
                    padding="same",
                    dilation_rate=c.get("d_factor", (1,1)),
                    use_bias=True,
                    activation=None,
                    name=c.get("name", "conv{}".format(i))))
            else:
                self.convs.add(layers.Conv2DTranspose(
                    c["chnl"],
                    kernel_size=c.get("ksize", (3,3)),
                    strides=c.get("stride", (2,2)),
                    padding="same",
                    dilation_rate=c.get("d_factor", (1,1)),
                    use_bias=True,
                    activation=None,
                    name=c.get("name", "conv{}".format(i))))
        
    def call(self, inp):
        return self.convs(inp)


class EdgeGenerator(keras.Model):
    def __init__(self, config, **kwargs):
        super(EdgeGenerator, self).__init__(**kwargs)        
        self.model = GatedConvGenerator(config, name="convolutions")
    
    def call(self, masked_gray, masked_edge, mask):
        inp = tf.concat((masked_gray, masked_edge, mask), axis=3)
        raw_pred = tf.sigmoid(self.model(inp))
        return raw_pred * (1-mask) + masked_edge


class InpaitingGenerator(keras.Model):
    def __init__(self, config, **kwargs):
        super(InpaitingGenerator, self).__init__(**kwargs)
        self.model = GatedConvGenerator(config, name="convolutions")
    
    def call(self, edge, masked_clr, mask):
        inp = tf.concat((edge, masked_clr, mask), axis=3)
        raw_pred = tf.tanh(self.model(inp))
        return raw_pred * (1-mask) + masked_clr

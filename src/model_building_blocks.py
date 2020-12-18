import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import numpy as np

class GatedConv2d(layers.Layer):
    def __init__(self, filter_number, ksize=(3,3), strides=(1,1), dilation_rate=(1,1), res=False, **kwargs):
        super(GatedConv2d, self).__init__(**kwargs)
        #initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.res = res
        self.conv = layers.Conv2D(
            filter_number * 2,
            kernel_size=ksize,
            strides=strides,
            padding="same",
            dilation_rate=dilation_rate,
            use_bias=True,
            activation=None
            #kernel_initializer=initializer
            )
        self.inl = tfa.layers.InstanceNormalization(
            axis=3,
            center=True,
            scale=True,
            beta_initializer="random_uniform",
            gamma_initializer="random_uniform")

    def call(self, prev):
        convolved = self.conv(prev)
        gating, feature = tf.split(convolved, 2, 3)
        normalized_feature = self.inl(feature)
        activate_feature = tf.nn.elu(normalized_feature)
        smooth_gating = tf.nn.sigmoid(gating)
        result = activate_feature * smooth_gating
        if self.res:
            result += prev
        # print(self.name, " ", tf.reduce_min(result), tf.reduce_max(result), tf.reduce_mean(result), tf.math.reduce_std(result))
        return result


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
        self.inl = tfa.layers.InstanceNormalization(
            axis=3,
            center=True,
            scale=True,
            beta_initializer="random_uniform",
            gamma_initializer="random_uniform")
    
    def call(self, prev):
        convolved = self.conv(prev)
        gating, feature = tf.split(convolved, 2, 3)
        normalized_feature = self.inl(feature)
        activate_feature = tf.nn.elu(normalized_feature)
        smooth_gating = tf.nn.sigmoid(gating)
        result = activate_feature * smooth_gating
        # print(self.name, " ", tf.reduce_min(result), tf.reduce_max(result), tf.reduce_mean(result), tf.math.reduce_std(result))
        return result


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
                        name=c.get("name", "conv{}".format(i)),
                        res=c.get("res", False)
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
                self.convs.add(
                    layers.Conv2D(
                        c["chnl"],
                        kernel_size=c.get("ksize", (3,3)),
                        strides=c.get("stride", (1,1)),
                        padding="same",
                        dilation_rate=c.get("d_factor", (1,1)),
                        use_bias=True,
                        activation=None,
                        name=c.get("name", "conv{}".format(i)),
                        #kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=3.0)
                    )
                )
            else:
                self.convs.add(
                    layers.Conv2DTranspose(
                        c["chnl"],
                        kernel_size=c.get("ksize", (3,3)),
                        strides=c.get("stride", (2,2)),
                        padding="same",
                        dilation_rate=c.get("d_factor", (1,1)),
                        use_bias=True,
                        activation=None,
                        name=c.get("name", "conv{}".format(i)),
                        #kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=3.0)
                    )
                )
        
    def call(self, inp):
        return self.convs(inp)

def gen_conv_layers(config):
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
    convs = []
    for i, c in enumerate(config):
        if c["mode"] == "gconv":
            convs.append(
                GatedConv2d(
                    c["chnl"],
                    ksize=c.get("ksize", (3,3)),
                    strides=c.get("stride", (1,1)),
                    dilation_rate=c.get("d_factor", (1,1)),
                    name=c.get("name", "conv{}".format(i)),
                    res=c.get("res", False)
                )
            )
        elif c["mode"] == "gdeconv":
            convs.append(
                GatedDeconv2d(
                    c["chnl"],
                    ksize=c.get("ksize", (3,3)),
                    strides=c.get("stride", (2,2)),
                    dilation_rate=c.get("d_factor", (1,1)),
                    name=c.get("name", "conv{}".format(i))
                )
            )
        elif c["mode"] == "conv":
            convs.append(
                layers.Conv2D(
                    c["chnl"],
                    kernel_size=c.get("ksize", (3,3)),
                    strides=c.get("stride", (1,1)),
                    padding="same",
                    dilation_rate=c.get("d_factor", (1,1)),
                    use_bias=True,
                    activation=None,
                    name=c.get("name", "conv{}".format(i)),
                    #kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=3.0)
                )
            )
        else:
            convs.append(
                layers.Conv2DTranspose(
                    c["chnl"],
                    kernel_size=c.get("ksize", (3,3)),
                    strides=c.get("stride", (2,2)),
                    padding="same",
                    dilation_rate=c.get("d_factor", (1,1)),
                    use_bias=True,
                    activation=None,
                    name=c.get("name", "conv{}".format(i)),
                    #kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=3.0)
                )
            )
    return convs


# class EdgeGenerator(keras.Model):
#     def __init__(self, config, **kwargs):
#         super(EdgeGenerator, self).__init__(**kwargs)        
#         self.model = GatedConvGenerator(config, name="convolutions")
    
#     def call(self, masked_gray, masked_edge, mask):
#         inp = tf.concat((masked_gray, masked_edge, mask), axis=3)
#         logit = self.model(inp)
#         # print("eg_before: ", tf.reduce_min(logit), tf.reduce_max(logit), tf.reduce_mean(logit), tf.math.reduce_std(logit))
#         raw_pred = tf.sigmoid(logit)
#         # print("eg_range: ", tf.reduce_min(raw_pred), tf.reduce_max(raw_pred), tf.reduce_mean(logit), tf.math.reduce_std(logit))
#         # return raw_pred * (1-mask) + masked_edge
#         # print(tf.reduce_min(raw_pred), tf.reduce_max(raw_pred))
#         return raw_pred

class EdgeGenerator(keras.Model):
    def __init__(self, config, **kwargs):
        super(EdgeGenerator, self).__init__(**kwargs)
        # self.model = GatedConvGenerator(config["layers"], name="convolutions")
        self.convs = gen_conv_layers(config["layers"])
        self.jumps = config["jump_connection"]
    
    def call(self, masked_gray, masked_edge, mask):
        curr = tf.concat((masked_gray, masked_edge, mask), axis=3)
        curr_result = []
        for c, j in zip(self.convs, self.jumps):
            curr = c(curr)
            curr_result.append(curr)
            if j != -1:
                curr = tf.concat((curr, curr_result[j]), axis=3)
        # logit = self.model(inp)
        # print("eg_before: ", tf.reduce_min(logit), tf.reduce_max(logit), tf.reduce_mean(logit), tf.math.reduce_std(logit))
        raw_pred = tf.sigmoid(curr)
        # print("eg_range: ", tf.reduce_min(raw_pred), tf.reduce_max(raw_pred), tf.reduce_mean(logit), tf.math.reduce_std(logit))
        # return raw_pred * (1-mask) + masked_edge
        # print(tf.reduce_min(raw_pred), tf.reduce_max(raw_pred))
        return raw_pred


class InpaitingGenerator(keras.Model):
    def __init__(self, config, **kwargs):
        super(InpaitingGenerator, self).__init__(**kwargs)
        self.convs = gen_conv_layers(config["layers"])
        self.jumps = config["jump_connection"]
    
    def call(self, edge, masked_clr, mask):
        curr = tf.concat((edge, masked_clr, mask), axis=3)
        curr_result = []
        for c, j in zip(self.convs, self.jumps):
            curr = c(curr)
            curr_result.append(curr)
            if j != -1:
                curr = tf.concat((curr, curr_result[j]), axis=3)
        raw_pred = tf.tanh(curr)
        #return raw_pred * (1-mask) + masked_clr
        return raw_pred

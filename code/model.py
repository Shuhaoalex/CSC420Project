import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class GatedConv2d(layers.Layer):
    def __init__(self, filter_number, ksize=(3,3), strides=(1,1), dialation_rate=(1,1), **kwargs):
        super(GatedConv2d, self).__init__(**kwargs)
        self.conv = layers.Conv2D(filter_number, ksize, strides, padding="same", use_bias=True, activation=None)
        
    def build(self, input_shape):
        input_layer = layers.Input(input_shape[1:])
        convolved = self.conv(input_layer)
        gating, feature = tf.split(convolved, 2, 3)
        activate_feature = tf.nn.elu(feature)
        smooth_gating = tf.nn.sigmoid(gating)
        result = activate_feature * smooth_gating
        self.gconv = keras.Model(inputs=input_layer, outputs=result, name="gated_conv")

    def call(self, prev):
        return self.gconv(prev)


class GatedDeconv2d(layers.Layer):
    def __init__(self, filter_number, ksize=(3,3), strides=(1,1), dialation_rate=(1,1), **kwargs):
        super(GatedDeconv2d, self).__init__(**kwargs)
        self.conv = GatedConv2d(filter_number, ksize, strides)
    
    def build(self, input_shape):
        input_layer = layers.Input(input_shape[1:])
        resized_input = tf.image.resize(input_layer, (input_shape[1] * 2, input_shape[2] * 2), method='nearest')
        result = self.conv(resized_input)
        self.gconv = keras.Model(inputs=input_layer, outputs=result, name="gated_conv")
    
    def call(self, prev):
        return self.gconv(prev)


class GatedConvGenerator(keras.Model):
    def __init__(self, config, **kwargs):
        """
        config is a list of dictionary containing parameter for each layer
        {
            "mode": "conv" or "deconv",
            "chnl": int,
            "ksize": (int, int),          --optional, default (3,3)
            "stride": (int, int),         --optional, default (1,1)
            "d_factor": (int, int)        --optional, default (1,1)
            "name": string                --optional, default conv_i
        }
        """
        super(GatedConvGenerator, self).__init__(**kwargs)
        self.convs = keras.Sequential()
        for i, c in enumerate(config):
            if c["mode"] == "conv":
                self.convs.add(
                    GatedConv2d(
                        c["chnl"],
                        ksize=c.get("ksize", (3,3)),
                        strides=c.get("stride", (1,1)),
                        dialation_rate=c.get("d_factor", (1,1)),
                        name=c.get("name", "conv{}".format(i))
                    )
                )
            else:
                self.convs.add(
                    GatedDeconv2d(
                        c["chnl"],
                        ksize=c.get("ksize", (3,3)),
                        strides=c.get("stride", (1,1)),
                        dialation_rate=c.get("d_factor", (1,1)),
                        name=c.get("name", "conv{}".format(i))
                    )
                )
    
    def build(self, input_shape):
        input_layer = layers.Input(input_shape[1:])
        self.model = keras.Sequential((input_layer, self.convs))
    
    def call(self, inp):
        return self.model(inp)


class EdgeGenerator(keras.Model):
    def __init__(self, config=None, **kwargs):
        super(EdgeGenerator, self).__init__(**kwargs)
        if config is None:
            cnum = 16
            config = [
                {"mode":"conv", "chnl": cnum, "ksize":(5,5), "name":"conv1"},
                {"mode":"conv", "chnl": 2*cnum, "stride":(2,2), "name":"conv2_downsample"},
                {"mode":"conv", "chnl": 2*cnum, "name":"conv3"},
                {"mode":"conv", "chnl": 4*cnum, "stride":(2,2), "name":"conv4_downsample"},
                {"mode":"conv", "chnl": 4*cnum, "name":"conv5"},
                {"mode":"conv", "chnl": 4*cnum, "d_factor":(2,2), "name":"conv6_astrous"},
                {"mode":"conv", "chnl": 4*cnum, "d_factor":(4,4), "name":"conv7_astrous"},
                {"mode":"conv", "chnl": 4*cnum, "d_factor":(8,8), "name":"conv8_astrous"},
                {"mode":"conv", "chnl": 4*cnum, "d_factor":(16,16), "name":"conv9_astrous"},
                {"mode":"conv", "chnl": 4*cnum, "name":"conv10"},
                {"mode":"deconv", "chnl": 2*cnum, "name":"conv11_upsample"},
                {"mode":"conv", "chnl": 2*cnum, "name":"conv12"},
                {"mode":"deconv", "chnl": cnum, "name":"conv13_upsample"},
                {"mode":"conv", "chnl": 1, "name":"conv14"},
            ]
        
        self.model = GatedConvGenerator(config, name="convolutions")
    
    def build(self, input_shape):
        input_layer = layers.Input(input_shape[1:])
        self.model = keras.Sequential((input_layer, self.convs))
    
    def call(self, inp):
        return self.model(inp)


class InpaitingGenerator(keras.Model):
    def __init__(self, config=None, **kwargs):
        super(InpaitingGenerator, self).__init__(**kwargs)
        if config is None:
            cnum = 32
            config = [
                {"mode":"conv", "chnl": cnum, "ksize":(5,5), "name":"conv1"},
                {"mode":"conv", "chnl": 2*cnum, "stride":(2,2), "name":"conv2_downsample"},
                {"mode":"conv", "chnl": 2*cnum, "name":"conv3"},
                {"mode":"conv", "chnl": 4*cnum, "stride":(2,2), "name":"conv4_downsample"},
                {"mode":"conv", "chnl": 4*cnum, "name":"conv5"},
                {"mode":"conv", "chnl": 4*cnum, "d_factor":(2,2), "name":"conv6_astrous"},
                {"mode":"conv", "chnl": 4*cnum, "d_factor":(4,4), "name":"conv7_astrous"},
                {"mode":"conv", "chnl": 4*cnum, "d_factor":(8,8), "name":"conv8_astrous"},
                {"mode":"conv", "chnl": 4*cnum, "d_factor":(16,16), "name":"conv9_astrous"},
                {"mode":"conv", "chnl": 4*cnum, "name":"conv10"},
                {"mode":"deconv", "chnl": 2*cnum, "name":"conv11_upsample"},
                {"mode":"conv", "chnl": 2*cnum, "name":"conv12"},
                {"mode":"deconv", "chnl": cnum, "name":"conv13_upsample"},
                {"mode":"conv", "chnl": 1, "name":"conv14"},
            ]
        
        self.model = GatedConvGenerator(config, name="convolutions")
    
    def build(self, input_shape):
        input_layer = layers.Input(input_shape[1:])
        self.model = keras.Sequential((input_layer, self.convs))
    
    def call(self, inp):
        return self.model(inp)
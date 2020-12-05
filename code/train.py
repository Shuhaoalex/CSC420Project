import tensorflow as tf
import numpy as np
from model import *
from losses import *
from dataset import Dataset

model_config = {
    "edge": {
        "lamb_adv": 1,
        "lamb_fm": 10,
        "generator" : [
            {"mode":"conv", "chnl": 16, "ksize":(5,5), "name":"conv1"},
            {"mode":"conv", "chnl": 32, "stride":(2,2), "name":"conv2_downsample"},
            {"mode":"conv", "chnl": 32, "name":"conv3"},
            {"mode":"conv", "chnl": 64, "stride":(2,2), "name":"conv4_downsample"},
            {"mode":"conv", "chnl": 64, "name":"conv5"},
            {"mode":"conv", "chnl": 64, "d_factor":(2,2), "name":"conv6_astrous"},
            {"mode":"conv", "chnl": 64, "d_factor":(4,4), "name":"conv7_astrous"},
            {"mode":"conv", "chnl": 64, "name":"conv10"},
            {"mode":"deconv", "chnl": 32, "name":"conv11_upsample"},
            {"mode":"conv", "chnl": 32, "name":"conv12"},
            {"mode":"deconv", "chnl": 16, "name":"conv13_upsample"},
            {"mode":"conv", "chnl": 1, "name":"conv14"},
        ]
    },
    "clr": {
        "lamb_l1": 1,
        "lamb_adv": 0.1,
        "lamb_perc": 0.1,
        "lamb_style": 250, # paper use 250 here
        "generator": [
            {"mode":"conv", "chnl": 32, "ksize":(5,5), "name":"conv1"},
            {"mode":"conv", "chnl": 64, "stride":(2,2), "name":"conv2_downsample"},
            {"mode":"conv", "chnl": 64, "name":"conv3"},
            {"mode":"conv", "chnl": 128, "stride":(2,2), "name":"conv4_downsample"},
            {"mode":"conv", "chnl": 128, "name":"conv5"},
            {"mode":"conv", "chnl": 128, "d_factor":(2,2), "name":"conv6_astrous"},
            {"mode":"conv", "chnl": 128, "d_factor":(4,4), "name":"conv7_astrous"},
            {"mode":"conv", "chnl": 128, "d_factor":(8,8), "name":"conv8_astrous"},
            {"mode":"conv", "chnl": 128, "d_factor":(16,16), "name":"conv9_astrous"},
            {"mode":"conv", "chnl": 128, "name":"conv10"},
            {"mode":"deconv", "chnl":64, "name":"conv11_upsample"},
            {"mode":"conv", "chnl": 64, "name":"conv12"},
            {"mode":"deconv", "chnl": 32, "name":"conv13_upsample"},
            {"mode":"conv", "chnl": 3, "name":"conv14"},
        ]
    }
}


dataset_config = {"img_train_flist":"../datasets/celeba_train.flist",
    "img_test_flist":"../datasets/celeba_test.flist",
    "img_validation_flist":"../datasets/celeba_validation.flist",
    "mask_train_flist":"../datasets/mask_train.flist",
    "mask_validation_flist":"../datasets/mask_validation.flist",
    "mask_test_flist":"../datasets/mask_test_test.flist",
    "sigma":2, "input_size":256
}


edge_generator = EdgeGenerator(config=model_config["edge"]["generator"], name="EdgeGenerator")
eg_opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

inpainting_generator = InpaitingGenerator(config=model_config["clr"]["generator"], name="InpaitingGenerator")
ig_opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

edge_discriminator = EdgeDiscriminator(name="EdgeDiscriminator")
ed_opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

inpainting_discriminator = InpaintingDiscriminator(name="InpaintingDiscriminator")
id_opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

perceptual_and_style_loss = PerceptuaAndStylelLoss(name="PL_Loss")


@tf.function
def edge_train_step(masked_gray_img, edge, mask):
    masked_edge = mask * edge
    gen_input = tf.concat((masked_edge, mask), axis=3)
    with tf.GradientTape(persistent=True) as tape:
        fake_edge = edge_generator(gen_input)
        adv_loss, lfm = edge_discriminator.generator_loss(fake_edge, edge)
        gen_loss = model_config["edge"]["lamb_adv"] * adv_loss + model_config["edge"]["lamb_fm"] * lfm
        disc_loss = edge_discriminator.discriminator_loss(fake_edge, edge)
    
    gen_grad = tape.gradient(gen_loss, edge_generator.trainable_variables)
    disc_grad = tape.gradient(disc_loss, edge_discriminator.trainable_variables)

    eg_opt.apply_gradients(zip(gen_grad, edge_generator.trainable_variables))
    ed_opt.apply_gradients(zip(disc_grad, edge_discriminator.trainable_variables))


@tf.function
def inpainting_train_step(edge, clr_img, mask):
    masked_clr = mask * clr_img
    gen_input = tf.concat((edge, masked_clr, mask), axis=3)
    with tf.GradientTape(persistent=True) as tape:
        fake_clr = inpainting_generator(gen_input)
        adv_loss = inpainting_discriminator.generator_loss(fake_clr, clr_img)
        perc_loss, style_loss = perceptual_and_style_loss(fake_clr, clr_img)
        l1_loss = reconstruction_loss(fake_clr, clr_img)
        gen_loss = model_config["clr"]["lamb_l1"] * l1_loss +\
                   model_config["clr"]["lamb_adv"] * adv_loss +\
                   model_config["clr"]["lamb_perc"] * perc_loss +\
                   model_config["clr"]["lamb_style"] * style_loss
        disc_loss = inpainting_discriminator.discriminator_loss(fake_clr, clr_img)
    
    gen_grad = tape.gradient(gen_loss, inpainting_generator.trainable_variables)
    disc_grad = tape.gradient(disc_loss, inpainting_discriminator.trainable_variables)

    ig_opt.apply_gradients(zip(gen_grad, inpainting_generator.trainable_variables))
    id_opt.apply_gradients(zip(disc_grad, inpainting_discriminator.trainable_variables))




dataset = Dataset(dataset_config, dataset_config["img_train_flist"], dataset_config["mask_train_flist"], training=True)
dataset = dataset.dataset.shuffle(100).batch(10).prefetch(200)


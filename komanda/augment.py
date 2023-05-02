import pandas as pd
from pathlib import Path
import os
import tensorflow as tf
import tensorflow_addons as tfa
from random import randrange as rr
from random import uniform as ru
#tf.compat.v1.disable_v2_behavior()
#tf.compat.v1.disable_eager_execution()
abs_path = "C:\\Users\\Pratham B\\Desktop\\UNI_Y3\\PRBX\\data\\data\\hmb1\\"#local
new_path = "C:\\Users\\Pratham B\\Desktop\\UNI_Y3\\PRBX\\augment_py\\" #local
filename = Path("C:/Users/Pratham B/Desktop/UNI_Y3/PRBX/augment_py")

rad1 = -0.78
rad2 = 0.78
factor1 = -0.5
factor2 = 0.25
box1 = 64
box2 = 128
#zoom_range = (ru(0.0, 2.5), ru(0.0, 2.5))
zoom1 = (0.1, 0.1)
zoom2 = (2.5, 2.5)
#intensity = ru(0.0, 200.0)
intens1 = 100.0
intens2 = 200.0
#delta = ru(-0.5, 0.5)
delta1 = -0.5
delta2 = 0.5

df_train = pd.read_csv("interpolated_test.csv", header=None)
lst = df_train[5].to_list()

def augmentations(i):
    tf.keras.utils.save_img(os.path.join(filename, "raw.jpg"), i.numpy())

    output = tfa.image.rotate(i, rad1)
    tf.keras.utils.save_img(os.path.join(filename, "rotate1.jpg"), output.numpy())
    output = tfa.image.rotate(i, rad2)
    tf.keras.utils.save_img(os.path.join(filename, "rotate2.jpg"), output.numpy())
    
    output = tf.convert_to_tensor(tf.keras.preprocessing.image.random_shear(i, intens1), tf.float64)
    tf.keras.utils.save_img(os.path.join(filename, "shear1.jpg"), output.numpy())
    output = tf.convert_to_tensor(tf.keras.preprocessing.image.random_shear(i, intens2), tf.float64)
    tf.keras.utils.save_img(os.path.join(filename, "shear2.jpg"), output.numpy())

    output = tf.convert_to_tensor(tf.keras.preprocessing.image.random_zoom(i, zoom1), tf.float64)
    tf.keras.utils.save_img(os.path.join(filename, "zoom1.jpg"), output.numpy())
    output = tf.convert_to_tensor(tf.keras.preprocessing.image.random_zoom(i, zoom2), tf.float64)
    tf.keras.utils.save_img(os.path.join(filename, "zoom2.jpg"), output.numpy())

    output = tf.image.adjust_hue(i, factor1)
    tf.keras.utils.save_img(os.path.join(filename, "hue1.jpg"), output.numpy())
    output = tf.image.adjust_hue(i, factor2)
    tf.keras.utils.save_img(os.path.join(filename, "hue2.jpg"), output.numpy())

    output = tf.squeeze(tfa.image.random_cutout(tf.expand_dims(i, axis=0), (box1, box1)))
    tf.keras.utils.save_img(os.path.join(filename, "occ1.jpg"), output.numpy())
    output = tf.squeeze(tfa.image.random_cutout(tf.expand_dims(i, axis=0), (box2, box2)))
    tf.keras.utils.save_img(os.path.join(filename, "occ2.jpg"), output.numpy())

    output = tf.clip_by_value(tf.image.adjust_brightness(i, delta1), 0.0, 1.0)
    tf.keras.utils.save_img(os.path.join(filename, "bri1.jpg"), output.numpy())
    output = tf.clip_by_value(tf.image.adjust_brightness(i, delta2), 0.0, 1.0)
    tf.keras.utils.save_img(os.path.join(filename, "bri2.jpg"), output.numpy())

    print ("ok")

img_raw = tf.io.read_file(abs_path + str(lst[0]).replace("/", "\\"))
img = tf.io.decode_image(img_raw)
img = tf.image.convert_image_dtype(img, tf.float64)
augmentations(img)
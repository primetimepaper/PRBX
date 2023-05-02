import pandas as pd
from pathlib import Path
import os
import tensorflow as tf
import tensorflow_addons as tfa
from random import randrange as rr
from random import uniform as ru
#tf.compat.v1.disable_v2_behavior()
#tf.compat.v1.disable_eager_execution()
abs_path = "C:\\Users\\Pratham B\\Desktop\\UNI_Y3\\PRBX\\data\\data\\hmb1\\" #local
new_path = "C:\\Users\\Pratham B\\Desktop\\UNI_Y3\\PRBX\\github\\hmb1_L2\\" #local
#abs_path = "/shared/storage/cs/studentscratch/pb1028/new_venv/hmb1/"  #gpu
#new_path = "/shared/storage/cs/studentscratch/pb1028/new_venv/hmb1_L1/"  #gpu


""" other ops
tf.image - adjust_contrast, adjust_gamma, adjust_saturation
"""

# probabilities 1 - 40% chance to be augmented - equal probability for each type of augmentation - p = (0.4/6)
p = 0.4
# probabilities 2 - 50% chance to be augmented - equal probability for each type of augmentation - p = (0.5/6)
#p = 0.5
# probabilities 3 - 75% chance to be augmented - equal probability for each type of augmentation - p = (0.75/6)
#p = 0.75

def rand_L2(input_images, probs, prev_op=6, countdown=0.0, f=0, val=None, zoom=None): # augments a list of images with the same transformation (only 1 augmentation per list of images)
    output_images = []
    op = rr(6)
    if (prev_op == 6) and (countdown == 0.0) and ru(0.0, 1.0) < probs:
        countdown = 0.1
        rads = ru(-0.78, 0.78) # val/v
        factor = ru(-1.0, 1.0) # val/v
        box = rr(128) # val/v
        while box % 2 != 0:
            box = rr(128)
        zoom_range = (ru(0.0, 2.5), ru(0.0, 2.5)) # zoom/z
        intensity = ru(0.0, 150.0) # val/v
        delta = ru(-0.5, 0.5) # val/v
        v = None
        z = None
        for i in input_images:
            if op == 0:
                # rotate - correlates between cams
                output_images.append(tfa.image.rotate(i, rads*countdown))
                v = rads
            elif op == 1:
                #shear - does not correlate
                output_images.append(tf.convert_to_tensor(tf.keras.preprocessing.image.random_shear(i, intensity*countdown), tf.float64))
                v = intensity
            elif op == 2:
                #zoom - does not correlate
                output_images.append(tf.convert_to_tensor(tf.keras.preprocessing.image.random_zoom(i, tuple([countdown*x for x in zoom_range])), tf.float64))
                z = zoom_range
            elif op == 3:
                #hue - correlates
                output_images.append(tf.image.adjust_hue(i, factor*countdown))
                v = factor
            elif op == 4:
                #occlude - does not correlate, no gradient increase/constant across all frames
                output_images.append(tf.squeeze(tfa.image.random_cutout(tf.expand_dims(i, axis=0), (box, box))))
                v = box
            elif op == 5:
                #brightness - correlates
                output_images.append(tf.clip_by_value(tf.image.adjust_brightness(i, delta*countdown), 0.0, 1.0))
                v = delta
            else:
                output_images.append(i)
        return output_images, op, (countdown + 0.1), v, z, 2
    elif countdown != 0.0 and prev_op != 6:
        for i in input_images:
            if prev_op == 0:
                # rotate - correlates between cams
                output_images.append(tfa.image.rotate(i, val*countdown))
            elif prev_op == 1:
                #shear - does not correlate
                output_images.append(tf.convert_to_tensor(tf.keras.preprocessing.image.random_shear(i, val*countdown), tf.float64))
            elif prev_op == 2:
                #zoom - does not correlate
                output_images.append(tf.convert_to_tensor(tf.keras.preprocessing.image.random_zoom(i, tuple([countdown*x for x in zoom])), tf.float64))
            elif prev_op == 3:
                #hue - correlates
                output_images.append(tf.image.adjust_hue(i, val*countdown))
            elif prev_op == 4:
                #occlude - does not correlate, no gradient increase/constant across all frames
                output_images.append(tf.squeeze(tfa.image.random_cutout(tf.expand_dims(i, axis=0), (val, val))))
            elif prev_op == 5:
                #brightness - correlates
                output_images.append(tf.clip_by_value(tf.image.adjust_brightness(i, val*countdown), 0.0, 1.0))
            else:
                output_images.append(i)
        if f > 19:
            return output_images, 6, 0.0, None, None, 0
        elif f < 10:
            return output_images, prev_op, (countdown + 0.01), val, zoom, f+1
        else:
            return output_images, prev_op, (countdown - 0.01), val, zoom, f+1
    else:
        return input_images, 6, 0.0, None, None, 0

#df = pd.read_csv("interpolated.csv")
#df_train = pd.read_csv("interpolated_train.csv", header=None)
df_train = pd.read_csv("interpolated_test.csv", header=None)

lst = df_train[5].to_list()

idx = 0
counter = 1
inputs = []
names = []
outputs = []
prev = 6
c = 0.0
frame = 0
value = None
zoom = None

filename = Path("C:/Users/Pratham B/Desktop/UNI_Y3/PRBX/github/hmb1_L2")
def write(outputs, names):
    for i in range(len(names)):
        print(new_path + names[i])
        #tf.keras.utils.save_img((new_path + names[i]), outputs[i].numpy())
        tf.keras.utils.save_img(os.path.join(filename, names[i]), outputs[i].numpy())

while idx < len(lst):
    name = str(lst[idx]).replace("/", "\\") #str(lst[idx])
    names.append(name)
    img_raw = tf.io.read_file(abs_path + str(lst[idx]).replace("/", "\\")) #tf.io.read_file(abs_path + str(lst[idx]))
    img = tf.io.decode_image(img_raw)
    img = tf.image.convert_image_dtype(img, tf.float64)
    inputs.append(img)
    if idx > 0 and counter == 3:
        out, prev, c, value, zoom, frame = rand_L2(inputs, p, prev, c, frame, value, zoom)
        outputs += out
        write(outputs, names)
        outputs = []
        names = []
        inputs = []
        counter = 0
    idx += 1
    counter += 1
# randL2 params: input_images, probs, prev_op=6, countdown=0.0, f=0, val=None, zoom=None
# randL2 return: output_images, prev_op, (countdown +- 0.01), val, zoom, f
# save tensors as images to folder
if len(names) != len(outputs):
    print("lst LEN ERROR")
else:
    print("saving images to " + new_path + " ...")
"""
for i in range(len(names)):
    filename = Path("C:/Users/Pratham B/Desktop/UNI_Y3/PRBX/github/hmb1_L2")
    print(new_path + names[i])
    #tf.keras.utils.save_img((new_path + names[i]), outputs[i].numpy())
    tf.keras.utils.save_img(os.path.join(filename, names[i]), outputs[i].numpy())
print("OK")
"""
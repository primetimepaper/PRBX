import os
import numpy as np
from PIL import Image
import math

#tren_sz = 0.6
#test_sz = 0.2
#vali_sz = 0.2
tren_sz = 0.1
test_sz = 0.05
vali_sz = 0.05
#csv_path = os.path.join('C:/Users/Pratham B/Desktop/UNI_Y3/PRBX/github/PRBX/PRBX/chauffeur/data/', 'labels.csv')
csv_path = os.path.join('chauffeur/data/', 'labels.csv')

labels = np.genfromtxt(csv_path, delimiter=",")[1:,1]
np.save('chauffeur/data/labels.npy', labels)
#labels = data = np.load('labels.npy')

# train/test/validation indexes as .npy
n = len(labels)
x = math.floor(n*tren_sz)
y = math.floor(n*test_sz)
z = math.floor(n*vali_sz)

tren_i = np.arange(1,x)
test_i = np.arange(x,x+y)
vali_i = np.arange(x+y,x+y+z)

np.save('chauffeur/data/training_indexes.npy', tren_i)
np.save('chauffeur/data/testing_indexes.npy', test_i)
np.save('chauffeur/data/validation_indexes.npy', vali_i)

# images as .npy
directory = 'chauffeur/data/images_jpg'
# iterate over files in that directory
i = 1
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        img = Image.open(f)
        #name = 'chauffeur/data/images/' + filename + '.npy'
        name = 'chauffeur/data/images/' + str(i) + '.npy'
        np.save(name, np.asarray(img))
    else:
        print("some images in images_jpg are corrupted.")
    i += 1
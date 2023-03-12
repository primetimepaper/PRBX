# imports
import logging
import time
import os
import csv
from PIL import Image
import numpy as np
"""
from models import LstmModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

task_id = str(int(time.time()))
tmp_model_path = os.path.join('/tmp', '%s.h5' % task_id)
# Create LstmModel object
lstm_model_config = LstmModel.create(
    tmp_model_path,
    (10, 120, 320, 3),
    timesteps=10,
    W_l2=0.0001,
    scale=60.0)
task = {
    'task_id': task_id,
    'dataset_path': 'showdown_full',
    'final': True,
    'model_config': lstm_model_config,
    'training_args': {
        'pctl_sampling': 'uniform',
        'batch_size': 32,
        'epochs': 10,
    },
}
# np.arr of labels as .npy
"""
#folder_input = '/my_folder/input/'
csv_path = os.path.join('/data/', 'labels.csv')
p = os.path.join('data/', '')
print(p)
print(csv_path)
print(os.path.isfile('labels.csv'))
#file_content = open(os.path.join('./data/', "csv")).read()
labels = np.genfromtxt(csv_path, delimiter=",")
print(labels)
#labels_npy = np.save('labels.npy', labels)
# train/test/validation indexes as .npy
# images as .npy
# Create Dataset obj


# handle_task(task)
# LstmModel.fit
# save weights

# LstmModel.evaluate

print("OK")
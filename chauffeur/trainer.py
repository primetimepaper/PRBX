# imports
import logging
import time
import os
import csv
#from PIL import Image
import numpy as np
import math
from models import LstmModel
import models
import worker
import datasets

ind = '/shared/storage/cs/studentscratch/pb1028/new_venv/PRBX/chauffeur/'

#params
batch_sz = 32
epochs = 10

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

task_id = str(int(time.time()))
tmp_model_path = os.path.join('/tmp', '%s.h5' % task_id)
# Create LstmModel object
#(10, 120, 320, 3),
lstm_model_config = LstmModel.create(
    tmp_model_path,
    (10, 480, 640, 3),
    timesteps=10,
    W_l2=0.0001,
    scale=60.0)
task = {
    'task_id': task_id,
    'dataset_path': (ind+'data'),
    'final': True,
    'model_config': lstm_model_config,
    'training_args': {
        'pctl_sampling': 'uniform',
        'batch_size': batch_sz,
        'epochs': epochs,
    },
}
# np.arr of labels as .npy
#labels = data = np.load('labels.npy')
#print(labels)

# train/test/validation indexes as .npy
#np.arange(3,7) -> array([3, 4, 5, 6])
#n = len(labels)


# images as .npy
# Create Dataset obj


worker.handle_task(
    task, datasets_dir='ind', models_path=(ind+'models'))
# LstmModel.fit
# save weights

# LstmModel.evaluate

print("OK")
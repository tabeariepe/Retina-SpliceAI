###############################################################################
# This file contains code to test the SpliceAI model.
###############################################################################

import numpy as np
import sys
import time
import h5py
from keras.models import load_model
from utils import *
import argparse 
from spliceai import categorical_crossentropy_2d
import tensorflow_addons as tfa
import tensorflow as tf

# Ensure TensorFlow uses GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Read command line parameters 
parser = argparse.ArgumentParser()
parser.add_argument('model_name', type=str)
parser.add_argument('test_dataset', type=str)

# Parse the arguments
args = parser.parse_args()

name = args.model_name
dataset = args.test_dataset

###############################################################################
# Load model and test data
###############################################################################

BATCH_SIZE = 6
CL = 10000

version = [1, 2, 3, 4, 5]

model = [[] for v in range(len(version))]

for v in range(len(version)):

    model[v] = load_model('../models/' + name + '_' + str(version[v]) + '.h5', compile = False)

    model_architecture = name.split('_')[1]

    if model_architecture == 'optimized':
         optimizer = tfa.optimizers.AdamW(weight_decay=0.00001)
         model[v].compile(loss=categorical_crossentropy_2d, optimizer=optimizer)
    else:
        model[v].compile(loss=categorical_crossentropy_2d, optimizer='adam')
    
    print(model)
      
# Load the testdata
h5f = h5py.File('/data/' + dataset + '_test_0.h5', 'r')

num_idx = len(list(h5f.keys()))//2

print(dataset)

###############################################################################
# Model testing
###############################################################################

start_time = time.time()

output_class_labels = ['Null', 'Acceptor', 'Donor']
# The three neurons per output correspond to no splicing, splice acceptor (AG)
# and splice donor (GT) respectively.

for output_class in [1, 2]:

    Y_true = [[] for t in range(1)]
    Y_pred = [[] for t in range(1)]

    for idx in range(num_idx):

        X = h5f['X' + str(idx)][:]
        Y = h5f['Y' + str(idx)][:]

        Xc, Yc = clip_datapoints(X, Y, CL, 1)

        Yps = [np.zeros(Yc[0].shape) for t in range(1)]

        for v in range(len(version)):

            Yp = model[v].predict(Xc, batch_size=BATCH_SIZE)

            if not isinstance(Yp, list):
                Yp = [Yp]

            for t in range(1):
                Yps[t] += Yp[t]/len(version)
        # Ensemble averaging (mean of the ensemble predictions is used)

        for t in range(1):

            is_expr = (Yc[t].sum(axis=(1,2)) >= 1)

            Y_true[t].extend(Yc[t][is_expr, :, output_class].flatten())
            Y_pred[t].extend(Yps[t][is_expr, :, output_class].flatten())

    print("\n%s:" % (output_class_labels[output_class]))

    for t in range(1):

        Y_true[t] = np.asarray(Y_true[t])
        Y_pred[t] = np.asarray(Y_pred[t])

        print_topl_statistics(Y_true[t], Y_pred[t])


h5f.close()

print("--- %s seconds ---" % (time.time() - start_time))
print("--------------------------------------------------------------")

###############################################################################






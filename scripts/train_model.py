###############################################################################
# This file contains the code to train the SpliceAI model.
###############################################################################

import numpy as np
import sys
import time
import h5py
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
import tensorflow_addons as tfa
import argparse 
from tensorflow.keras.models import load_model
import gc
from tensorflow.keras.callbacks import Callback

class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session() 

print(tf.__version__)

###############################################################################
# Command line input
##############################################################################

parser = argparse.ArgumentParser(description='Description of your script.')

# Add arguments
parser.add_argument('model_number', type=int, help='Model number')
parser.add_argument('dataset', type=str, help='Dataset used for the training')
parser.add_argument('model_architecture', type=str, choices=['standard', 'dropout', 'optimized', 'freeze'], help='Model that is trained')
parser.add_argument('mode', type=str, choices=['initialize', 'train'], help='Training mode')
parser.add_argument('--dropoutrate', type=float, help='Dropout rate used during training')
parser.add_argument('--freezeoption', type=str, default='A', choices=['A', 'B', 'C','D','E','F'],
                    help='Determines how many layers are frozen when retraining the GTEx model.')

# Parse the arguments
args = parser.parse_args()

# Assign arguments to variables
model_number = args.model_number
mode = args.mode
dataset = args.dataset
model_architecture = args.model_architecture

# Now you can use these variables as needed in your script
print("Architecure:", model_architecture)
print("Model Number:", model_number)
print("Mode:", mode)
print("Dataset:", dataset)

###############################################################################
# Parameters
##############################################################################
CL_max=10000
SL=5000
num_epochs = 10

if model_architecture == 'dropout':
    from spliceai_dropout import *
    from utils import *
elif model_architecture in ['standard', 'freeze']:
    from spliceai import *
    from utils import *
elif model_architecture == 'optimized':
    from spliceai import *
    from utils_optimized import *


###############################################################################
# Model
###############################################################################
# Hyper-parameters:
# L: Number of convolution kernels
# W: Convolution window size in each residual unit
# AR: Atrous rate in each residual unit
L = 32
N_GPUS = 1

if int(CL_max) == 80:
    W = np.asarray([11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1])
    BATCH_SIZE = 18*N_GPUS
elif int(CL_max) == 400:
    W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
    BATCH_SIZE = 18*N_GPUS
elif int(CL_max) == 2000:
    W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                    21, 21, 21, 21])
    AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                     10, 10, 10, 10])
    BATCH_SIZE = 12*N_GPUS
elif int(CL_max) == 10000:
    W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                    21, 21, 21, 21, 41, 41, 41, 41])
    AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                     10, 10, 10, 10, 25, 25, 25, 25])
    BATCH_SIZE = 6*N_GPUS

CL = 2 * np.sum(AR*(W-1))
print("Context nucleotides: %d" % (CL))
print("Sequence length (output): %d" % (SL))

if model_architecture == 'dropout':
    model = SpliceAI(L, W, AR, args.dropoutrate)
else:
    model = SpliceAI(L, W, AR)

# Decie if the model should be initialised or a existing initialization should be used
if mode == 'initialize':
    print('Saving initalization')
    model.save('../models/SpliceAI_initialization_' + str(model_number) + '.h5')
elif mode == 'train':
    print('Loading existing model')

    if model_architecture == 'freeze':
        init_model = load_model('../models/SpliceAI_standard_gtex_' + str(model_number) + '.h5', compile=False)
        print('Freezing layers')
        
        # Define the options for freezing layers
        options = {
            'A': set(['conv1d_38', 'batch_normalization_32']),
            'B': set(['conv1d_38', 'batch_normalization_32', 'conv1d_37']),
            'C': set(['conv1d_38', 'batch_normalization_32', 'conv1d_37', 'batch_normalization_30', 'conv1d_35', 'batch_normalization_31', 'conv1d_36']),
            'D': set(['conv1d_38', 'batch_normalization_32', 'conv1d_37', 'batch_normalization_30', 'conv1d_35', 'batch_normalization_31', 'conv1d_36', 'batch_normalization_28', 'conv1d_33', 'batch_normalization_29', 'conv1d_34']),
            'E': set(['conv1d_38', 'batch_normalization_32', 'conv1d_37', 'batch_normalization_30', 'conv1d_35', 'batch_normalization_31', 'conv1d_36', 'batch_normalization_28', 'conv1d_33', 'batch_normalization_29', 'conv1d_34', 'batch_normalization_26', 'conv1d_31', 'batch_normalization_27', 'conv1d_32']),
            'F': set(['conv1d_38', 'batch_normalization_32', 'conv1d_37', 'batch_normalization_30', 'conv1d_35', 'batch_normalization_31', 'conv1d_36', 'batch_normalization_28', 'conv1d_33', 'batch_normalization_29', 'conv1d_34', 'batch_normalization_26', 'conv1d_31', 'batch_normalization_27', 'conv1d_32', 'batch_normalization_24', 'conv1d_29', 'batch_normalization_25', 'conv1d_30'])
        }

        # Determine which option should be used
        chosen_option = options.get(args.freezeoption, options['F'])

        for layer in model.layers:
            if layer.name not in chosen_option:
                layer.trainable = False
                
        total_trainable_params = sum(tf.keras.backend.count_params(p) for p in model.trainable_weights)
        total_non_trainable_params = sum(tf.keras.backend.count_params(p) for p in model.non_trainable_weights)
        total_params = total_trainable_params + total_non_trainable_params
        print("Total Trainable Parameters:", total_trainable_params)
        print("Total Non-Trainable Parameters:", total_non_trainable_params)
        print("Total Parameters:", total_params)

    else:
        init_model = load_model('../models/SpliceAI_initialization_' + str(model_number) + '.h5', compile=False)
        
    model.set_weights(init_model.get_weights())
    model.summary()
    sys.stdout.flush()

    ###############################################################################
    # Training and validation
    ###############################################################################
    print('Loading training data')
    sys.stdout.flush()

    h5f = h5py.File('../data/' + dataset + '_train_all.h5')

    num_idx = len(list(h5f.keys()))//2
    # Add a seed to always initialyze the model the same 
    idx_all = np.random.default_rng(seed=model_number).permutation(num_idx)

    idx_train = idx_all[:int(0.9*num_idx)]
    idx_valid = idx_all[int(0.9*num_idx):]

    if model_architecture in ['standard', 'dropout', 'freeze']:
        print('standard training and loss')
        model.compile(loss=categorical_crossentropy_2d, optimizer='adam', run_eagerly=True)
    elif model_architecture == 'optimized':
        print('optimized training and loss')
        initial_learning_rate = 0.0005
        lr_schedule = CosineDecayRestarts(
            initial_learning_rate,
            first_decay_steps=2 * len(idx_train),
            t_mul=2.0,
            m_mul=1.0,
            alpha=0.0
        )
        optimizer = tfa.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=0.0001)
        model.compile(loss=categorical_crossentropy_2d, optimizer=optimizer, run_eagerly=True)
    else: 
        print('Model architecture not known')

    EPOCH_NUM = num_epochs*len(idx_train)

    start_time = time.time()

    print(('start time: ', start_time))
    sys.stdout.flush()

    counter = 1

    for epoch_num in range(EPOCH_NUM):
        
        rng_idx = np.random.default_rng()  
        idx = rng_idx.choice(idx_train)

        X = h5f['X' + str(idx)][:]
        Y = h5f['Y' + str(idx)][:]

        Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS) 

        model.fit(tf.convert_to_tensor(Xc) , [tf.convert_to_tensor(y) for y in Yc] ,
                    batch_size=BATCH_SIZE, verbose=0, callbacks=ClearMemory())

        if (epoch_num+1) % len(idx_train) == 0:

            print(counter)
            counter += 1

            # Printing metrics (see utils.py for details)

            print("--------------------------------------------------------------")
            print("\nValidation set metrics:")

            Y_true_1 = [[] for t in range(1)]
            Y_true_2 = [[] for t in range(1)]
            Y_pred_1 = [[] for t in range(1)]
            Y_pred_2 = [[] for t in range(1)]

            for idx in idx_valid:

                X = h5f['X' + str(idx)][:]
                Y = h5f['Y' + str(idx)][:]

                Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS)

                # check if nan in Xc
                Yp = model.predict(tf.convert_to_tensor(Xc), batch_size=BATCH_SIZE, verbose=0)

                if not isinstance(Yp, list):
                    Yp = [Yp]

                for t in range(1):

                    is_expr = (Yc[t].sum(axis=(1,2)) >= 1)

                    Y_true_1[t].extend(Yc[t][is_expr, :, 1].flatten())
                    Y_true_2[t].extend(Yc[t][is_expr, :, 2].flatten())
                    Y_pred_1[t].extend(Yp[t][is_expr, :, 1].flatten())
                    Y_pred_2[t].extend(Yp[t][is_expr, :, 2].flatten())

                # clean up
               # _ = gc.collect()
                #tf.keras.backend.clear_session()

            print("\nAcceptor:")
            for t in range(1):
                print_topl_statistics(np.asarray(Y_true_1[t]),np.asarray(Y_pred_1[t]))

            print("\nDonor:")
            for t in range(1):
                print_topl_statistics(np.asarray(Y_true_2[t]), np.asarray(Y_pred_2[t]))

            print("\nTraining set metrics:")

            Y_true_1 = [[] for t in range(1)]
            Y_true_2 = [[] for t in range(1)]
            Y_pred_1 = [[] for t in range(1)]
            Y_pred_2 = [[] for t in range(1)]

            for idx in idx_train[:len(idx_valid)]:

                X = h5f['X' + str(idx)][:]
                Y = h5f['Y' + str(idx)][:]

                Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS)

                Yp = model.predict(tf.convert_to_tensor(Xc) , batch_size=BATCH_SIZE, verbose=0)

                if not isinstance(Yp, list):
                    Yp = [Yp]

                for t in range(1):

                    is_expr = (Yc[t].sum(axis=(1,2)) >= 1)

                    Y_true_1[t].extend(Yc[t][is_expr, :, 1].flatten())
                    Y_true_2[t].extend(Yc[t][is_expr, :, 2].flatten())
                    Y_pred_1[t].extend(Yp[t][is_expr, :, 1].flatten())
                    Y_pred_2[t].extend(Yp[t][is_expr, :, 2].flatten())

                # clean up
                #_ = gc.collect()
                #tf.keras.backend.clear_session() 

            print("\nAcceptor:")
            for t in range(1):
                print_topl_statistics(np.asarray(Y_true_1[t]),np.asarray(Y_pred_1[t]))

            print("\nDonor:")
            for t in range(1):
                print_topl_statistics(np.asarray(Y_true_2[t]),np.asarray(Y_pred_2[t]))

            # Learning rate decay
            if model_architecture in ['standard', 'dropout', 'freeze']:
                print("Learning rate: %.5f" % (model.optimizer.lr.numpy()))
                if (epoch_num + 1) >= 6 * len(idx_train):
                    model.optimizer.lr.assign(0.5 * model.optimizer.lr)

            elif model_architecture == 'optimized':
                current_learning_rate = lr_schedule(optimizer.iterations)
                print("Learning rate: %.5f" % (current_learning_rate.numpy()))

            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()

            print("--------------------------------------------------------------")
            sys.stdout.flush()
            model.save('../models/SpliceAI_' + model_architecture + '_' + dataset + '_' + str(model_number) + '.h5')
        
    h5f.close()
        
###############################################################################

###############################################################################
# This file has the functions necessary to create the SpliceAI model.
###############################################################################

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Conv1D, Cropping1D, BatchNormalization, Add
import numpy as np

def ResidualUnit(l, w, ar):
    def f(input_node):
        bn1 = BatchNormalization()(input_node)
        act1 = Activation('relu')(bn1)
        conv1 = Conv1D(l, w, dilation_rate=ar, padding='same')(act1)
        bn2 = BatchNormalization()(conv1)
        act2 = Activation('relu')(bn2)
        conv2 = Conv1D(l, w, dilation_rate=ar, padding='same')(act2)
        output_node = Add()([conv2, input_node])
        return output_node

    return f

def SpliceAI(L, W, AR):
    assert len(W) == len(AR)

    CL = 2 * np.sum(AR*(W-1))

    input0 = Input(shape=(None, 4))
    conv = Conv1D(L, 1)(input0)
    skip = Conv1D(L, 1)(conv)

    for i in range(len(W)):
        conv = ResidualUnit(int(L), int(W[i]), int(AR[i]))(conv) 


        if (((i+1) % 4 == 0) or ((i+1) == len(W))):
            dense = Conv1D(L, 1)(conv)
            skip = Add()([skip, dense])

    skip = Cropping1D(int(CL/2))(skip)

    output0 = [[] for t in range(1)]

    for t in range(1):
        bn3 = BatchNormalization()(skip)
        output0[t] = Conv1D(3, 1, activation='softmax')(bn3)
    
    model = Model(inputs=input0, outputs=output0)

    return model

def categorical_crossentropy_2d(y_true, y_pred):


    y_true_float = tf.cast(y_true, dtype=tf.float32)
    
    return -tf.reduce_mean(
        y_true_float[:, :, 0] * tf.math.log(y_pred[:, :, 0] + 1e-10)
        + y_true_float[:, :, 1] * tf.math.log(y_pred[:, :, 1] + 1e-10)
        + y_true_float[:, :, 2] * tf.math.log(y_pred[:, :, 2] + 1e-10)
    )


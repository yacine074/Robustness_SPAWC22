

#from packages import *
#from parameters import *

import sys
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda, Dropout
from tensorflow.keras import Input, Model
from tensorflow.python.ops import nn



def get_model_DF(X_train, f_loss, f_metrics, f_activation1, f_activation2, f_activation3, LR) :
  
  """

      Structure of DL model for DF.

      Parameters:
         X_train: Channel gain array.
         f_loss: Loss function.
         f_metrics: List of metrics.
         f_activation1: First activation function for the output.
         f_activation2: Second activation function for the output.
         LR : Learning rate.
         
      Returns:
        DL model for DF.
  """
  opt = tf.keras.optimizers.Adam(learning_rate = LR)
  inputs = Input(shape=(X_train.shape[1]))
  x = Dense(128, activation='relu')(inputs)
  x = Dense(256, activation='relu')(x)
  x = Dense(256, activation='relu')(x)
  x = Dense(256, activation='relu')(x)
  output1 = Dense(1, activation=f_activation1)(x)
  output2 = Dense(1, activation=f_activation2)(x)
  output3 = Dense(1, activation=f_activation3)(x)# Access
  merged = tf.keras.layers.Concatenate()([output1, output2, output3])
  model = Model(inputs=inputs, outputs=[merged])
  model.compile(loss=f_loss, optimizer=opt, metrics=f_metrics)
  model.summary()
  return model


def get_model_CF(X_train, f_loss, f_metrics, f_activation1, f_activation2, LR) :
    """

      Structure of DL model for DF.

      Parameters:
         X_train: Channel gain array.
         f_loss: Loss function.
         f_metrics: List of metrics.
         f_activation1: First activation function for the output.
         f_activation2: Second activation function for the output.
         LR : Learning rate.
         
      Returns:
        DL model for CF.
    """
    opt = tf.keras.optimizers.Adam(learning_rate = LR)
    inputs = Input(shape=(X_train.shape[1]))
    x = Dense(128, activation='relu')(inputs)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    output1 = Dense(1, activation=f_activation1)(x)
    output2 = Dense(1, activation=f_activation2)(x)
    merged = tf.keras.layers.Concatenate()([output1, output2])
    model = Model(inputs=inputs, outputs=[merged])
    model.compile(loss=f_loss, optimizer=opt, metrics=f_metrics)
    model.summary()
    return model









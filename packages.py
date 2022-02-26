"""
  this file contains the different useful python libraries.

"""
# yacine 09/09/2021
import sys
import os
import warnings
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gekko import GEKKO
import scipy
from scipy.optimize import minimize
from scipy.optimize import fmin_cobyla
from scipy.stats import rice, nakagami
from tempfile import TemporaryFile
#import seaborn as sns;
#sns.set_theme(color_codes=True)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn import datasets, linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda, Dropout
from tensorflow.keras import Input, Model
from tensorflow.python.ops import nn

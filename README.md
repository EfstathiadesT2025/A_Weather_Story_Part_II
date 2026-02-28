# A_Weather_Story_Part_II
ClimateWins, a European nonprofit organization, aims to categorize and predict the weather in mainland Europe, while being concerned with the increase in extreme weather events, especially within the past 10 to 20 years.

# Objective - Goals
The objective of this project is to help predict the consequences of climate change, while using machine learning;
- Identifying weather patterns outside the regional norm in Europe
- Determining if unusual weather patterns are increasing
- Generating possibilities for future weather conditions over the next 25 to 50 years based on current trends
- Determining the safest places for people to live in Europe over the next 25 to 50 years

# Data Set
The data is collected by the European Climate Assessment & Data Set project (ECA&D), which recieved support
from the EUMETNET and the European Commission. EUMETNET is a collaborative network comprising
33 European National Meteorological Services.
Moreover, the data is based on weather observations from 18 different weather stations across Europe,
with data ranging from the late 1800s to 2022. Recordings exist for almost every day with values such as
temperature, wind speed, snow, global radiation and more.

The following 2 data sets were used for the training of the machine learning models:

- Dataset weather prediction processed
- Dataset answers pleasant weather

- # Tools
For this project, the following python libraries were used:

- import pandas as pd
- import numpy as np
- import seaborn as sns
- import os
- import operator
- import matplotlib.pyplot as plt
- import matplotlib.ticker as ticker
- from sklearn.preprocessing import StandardScaler
- from sklearn.metrics import confusion_matrix
- from sklearn.decomposition import PCA
- from matplotlib.pyplot import figure
- from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
- import pandas as pd
- import tensorflow as tf
- from sklearn import datasets
- from sklearn.model_selection import train_test_split
- from sklearn.ensemble import RandomForestClassifier
- from numpy import argmax
- from sklearn import metrics  
- from sklearn.tree import plot_tree
- from sklearn import tree
- from sklearn.preprocessing import LabelEncoder
- from numpy import unique
- from numpy import reshape
- from tensorflow.keras.models import Sequential
- from tensorflow.keras.layers import Conv1D, Conv2D, Dense, Dropout, BatchNormalization, Flatten, MaxPooling1D, TimeDistributed
- from tensorflow.keras.utils import to_categorical
- from sklearn import datasets
- from scipy.stats import randint
- from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
- from sklearn.metrics import confusion_matrix
- from tensorflow.keras.models import Sequential
- from tensorflow.keras.layers import MaxPooling2D, Activation
- from tensorflow.keras.datasets import mnist
- from PIL import Image
- from numpy import asarray
- import time
- import warnings
- warnings.filterwarnings("ignore")
- pd.set_option("display.max_columns", None)
- from sklearn.model_selection import (cross_val_score, StratifiedKFold)
- from sklearn.metrics import make_scorer, accuracy_score
- from tensorflow.keras.layers import LeakyReLU
- from tensorflow.keras.optimizers import (Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl)
- from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
- from scikeras.wrappers import KerasClassifier
- from bayes_opt import BayesianOptimization
- from math import floor
- from sklearn.metrics import ConfusionMatrixDisplay
- from tensorflow.keras import Input
- from tensorflow.keras.models import Model
- from tensorflow.keras.preprocessing.image import ImageDataGenerator
- %matplotlib inline
- from IPython.display import clear_output
- from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Reshape
- from tensorflow.keras.models import Model
- from tensorflow.keras.layers.convolutional import Conv2DTranspose

# Executing the Code
The code is available as jupyter notebooks, the scripts can be found in the [Supervised M](url) and [Unsupervised ML]() folders

# Final Presentation
Access the final presentation for this project [here](https://)

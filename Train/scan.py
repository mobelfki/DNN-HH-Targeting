#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from models import ClassifierForScan
from DataProcessing import Data
from keras import optimizers
from keras.models import model_from_json
from sklearn.metrics import roc_curve, roc_auc_score, auc
from scipy import interp
from itertools import cycle
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import train_test_split
from losses import Metrics, classifier_loss, oneplus_cosine_proximity, Roc
from focal_losses import categorical_focal_loss
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import sys
import keras
np.set_printoptions(threshold=sys.maxsize)

import os
import math
import ROOT
from ROOT import *
import root_numpy as nr
seed = 315
np.random.seed(seed)

data = Data(1);

inputs = data.X_Train.shape[1];

def model_clf(n_inputs=inputs,n_outputs=4,drop=0.05,neurons=128,activation='relu',init=keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='uniform')):

	model = Sequential()
	model.add(Dense(neurons, input_dim=n_inputs, activation=activation, kernel_initializer=init));		
	model.add(Dense(neurons, activation=activation, kernel_initializer=init));
	model.add(Dropout(drop));
	model.add(Dense(neurons, activation=activation, kernel_initializer=init));
	model.add(Dense(neurons, activation=activation, kernel_initializer=init));
	model.add(Dropout(drop));
	model.add(Dense(n_outputs, activation='softmax', kernel_initializer=init));
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy']);
	return model

activation = ['relu','elu']
neurons = [64,92,112,128,160,256]
batch_size = [100,500,10000,20000]
drop = [0.01,0.05,0.08,0.1,0.15,0.2]
loss = ['categorical_crossentropy',[categorical_focal_loss(alpha=1., gamma=0.)]]

param_grid = dict(activation=activation)

classweight = class_weight.compute_sample_weight("balanced",data.Y_Train)

model = KerasClassifier(build_fn=model_clf, epochs=2, verbose=1, shuffle=True, class_weight="balanced")

my_scorer = make_scorer(f1_score, greater_is_better=True, average='micro')

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, scoring=my_scorer)

grid_result = grid.fit(data.X_Train, data.Y_Train)


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



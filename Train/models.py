#!/usr/bin/env python
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization, AlphaDropout, GaussianNoise
import keras
import numpy as np

#Classifier class 

class Classifier():
	
	"""
	Classifier class
	
	Args: 
		name : name of the classifier
		n_inputs : number of inputs features
		n_outputs : number of outputs
		drop_rate : dropout rate
	"""

	def __init__(self, name, n_inputs, n_outputs, drop_rate, n_neuros):
		
		self.name = name;
		self.NInt = n_inputs;
		self.NOut = n_outputs;
		self.nN   = n_neuros;
		self.DropRate = drop_rate;

		self.Sequential = Sequential()
		self.Sequential.add(BatchNormalization(input_shape=(self.NInt,)));		
		self.Sequential.add(Dense(self.nN, input_dim=self.NInt, activation='relu', bias_initializer="zeros", kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform')));
		self.Sequential.add(Dense(self.nN, activation='relu', bias_initializer="zeros", kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform')));
		#self.Sequential.add(GaussianNoise(.2));		
		self.Sequential.add(Dropout(self.DropRate));
		self.Sequential.add(Dense(self.nN, activation='relu', bias_initializer="zeros", kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform')));
		#self.Sequential.add(GaussianNoise(.2));		
		self.Sequential.add(Dropout(self.DropRate));
		self.Sequential.add(Dense(self.nN, activation='relu', bias_initializer="zeros", kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform')));
		#self.Sequential.add(GaussianNoise(.2));
		self.Sequential.add(Dropout(self.DropRate));
		self.Sequential.add(Dense(self.nN, activation='relu', bias_initializer="zeros", kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform')));
		#self.Sequential.add(GaussianNoise(.2));		
		self.Sequential.add(Dropout(self.DropRate));
		self.Sequential.add(Dense(self.nN, activation='relu', bias_initializer="zeros", kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform')));
		#self.Sequential.add(GaussianNoise(.2));		
		self.Sequential.add(Dropout(self.DropRate));
		self.Sequential.add(Dense(self.NOut, activation='softmax', bias_initializer="zeros", kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform')));


class ClassifierForScan():
	
	"""
	Classifier class
	
	Args: 
		name : name of the classifier
		n_inputs : number of inputs features
		n_outputs : number of outputs
		drop_rate : dropout rate
	"""

	def __init__(self, name, n_inputs, n_outputs, drop_rate, n_neuros,activ,init):
		
		self.name = name;
		self.NInt = n_inputs;
		self.NOut = n_outputs;
		self.nN   = n_neuros;
		self.DropRate = drop_rate;

		self.Sequential = Sequential()
		self.Sequential.add(Dense(self.nN, input_dim=self.NInt, activation=activ, kernel_initializer=init));		
		self.Sequential.add(Dense(self.nN, activation=activ, kernel_initializer=init));
		self.Sequential.add(Dropout(self.DropRate));
		self.Sequential.add(Dense(self.nN, activation=activ, kernel_initializer=init));
		self.Sequential.add(Dense(self.nN, activation=activ, kernel_initializer=init));
		self.Sequential.add(Dropout(self.DropRate));
		self.Sequential.add(Dense(self.NOut, activation='softmax', kernel_initializer=init));
		

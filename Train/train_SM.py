#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from models import Classifier
from DataProcessing import Data
from keras import optimizers
from keras.models import model_from_json
from sklearn.metrics import roc_curve, roc_auc_score, auc
from scipy import interp
from itertools import cycle
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from losses import Metrics, classifier_loss, oneplus_cosine_proximity, Roc
from focal_losses import categorical_focal_loss
import itertools
import sys
import keras
np.set_printoptions(threshold=sys.maxsize)

import os
import math
seed = 215
np.random.seed(seed)

data = Data(1,'LO');

def getArgs():
	"""
	Get arguments from command line.
	"""
	args = ArgumentParser(description="Argumetns for training for DNN")
	args.add_argument('-n', '--nepochs', action='store', default=200, help='Number of epochs')
	args.add_argument('-o', '--noutput', action='store', default=4, help='Number of outputs from the model nclass')
	args.add_argument('-l', '--learning_rate', action='store', default=0.0001, help='learning rate')
	args.add_argument('-d', '--dropout_rate', action='store', default=0.15, help='Dropout rate') #0.13
	args.add_argument('-b', '--batch_size', action='store', default=1000, help='batch size')
	args.add_argument('-N', '--nerouns', action='store', default=128, help='number of neurons in each layer')
	return args.parse_args()

def train(N_epochs,n_outputs,drop,learing_rate,_batch_size,_neurons):
	
	n_inputs = data.X_Train.shape[1];

	clf = Classifier('DNN_Classifier',n_inputs,n_outputs,drop,_neurons).Sequential

	adm = optimizers.Adam(lr=learing_rate)

	clf.compile(loss=['categorical_crossentropy'], optimizer=adm, metrics=['categorical_accuracy']);

	#clf.compile(loss=categorical_focal_loss(gamma=2., alpha=.25), optimizer=adm, metrics=['categorical_accuracy']);

	es2       = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', min_delta=0.00001)
	save_best = ModelCheckpoint('weights/best.weights', monitor='val_loss', verbose=1, save_best_only=True)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=1, min_lr=0., verbose=1, mode='min', min_delta=0.0001)
	metrics   = Metrics()
	roc = Roc()
	callbacks_list = [es2,save_best]
	print(clf.summary())

	X = np.append(data.X_Train, data.X_Val, axis=0)
	Y = np.append(data.Y_Train, data.Y_Val, axis=0)
	W = np.append(data.W_Train, data.W_Val, axis=0)

	X_Train, X_Val, Y_Train, Y_Val, MC_Train, MC_Val = train_test_split(X, Y, W, test_size=0.33, random_state=215)

	Train_W = class_weight.compute_sample_weight('balanced', Y_Train)
	Val_W = class_weight.compute_sample_weight('balanced', Y_Val)
	
	history = clf.fit(X_Train, Y_Train, epochs=N_epochs, batch_size=_batch_size, shuffle=True, callbacks=callbacks_list, validation_data=(X_Val, Y_Val, Val_W), sample_weight=Train_W);

	del clf;

	return history

def plot_roc_curve(Y_true, Y_flase,dirname):
 
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(4):
    		fpr[i], tpr[i], _ = roc_curve(Y_true[:, i], Y_flase[:, i])
    		roc_auc[i] = auc(fpr[i], tpr[i])
	fpr["micro"], tpr["micro"], _ = roc_curve(Y_true.ravel(), Y_flase.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))

	mean_tpr = np.zeros_like(all_fpr)
	for i in range(4):
    		mean_tpr += interp(all_fpr, fpr[i], tpr[i])
	mean_tpr /= 4

	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

	plt.figure()
	plt.plot(fpr["micro"], tpr["micro"],
        label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)

	plt.plot(tpr["macro"], fpr["macro"],
        label='macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]),
        color='navy', linestyle=':', linewidth=4)
	colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
	samples = ['HH','ttH','ZH','YY']
	for i, color in zip(range(4), colors):
    		plt.plot(tpr[i], 1-fpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'''.format(samples[i], roc_auc[i]))

	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('Signal Eff')
	plt.ylabel('Bkg Rejection')
	plt.title('Some extension of Receiver operating characteristic to multi-class')
	plt.legend(loc="lower right")
	plt.savefig(dirname+"/ROC.png")

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, dirc='None'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(dirc+'/cm_SM.png')


def plot_model_history(model_history,dirname):

    fig, axs = plt.subplots(1,2,figsize=(15,5))

    axs[0].plot(range(1,len(model_history.history['categorical_accuracy'])+1),model_history.history['categorical_accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_categorical_accuracy'])+1),model_history.history['val_categorical_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['categorical_accuracy'])+1),len(model_history.history['categorical_accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')

    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig(dirname+"/LOSS.png")

def plot_Z(HH,ttH,ZH,JJ,cat):

	Scale_HH  = [4., 4., 4., 4., 4.];
	Scale_ZH  = [4., 4., 4., 4., 4.];
	Scale_ttH = [4., 4., 4., 4., 4.];
	Scale_JJ  = [4., 4., 4., 4., 4.]; 

#	pHH_HH  = HH[:,0]
#	pttH_HH = ttH[:,0]
#	pZH_HH  = ZH[:,0]
#	pJJ_HH  = JJ[:,0]

	pHH_HH  = np.array(HH)
	pttH_HH = np.array(ttH)
	pZH_HH  = np.array(ZH)
	pJJ_HH  = np.array(JJ)


	pHH_HH = pHH_HH[data.mask_HH];
	Y_HH = data.Y_HH[data.mask_HH];
	Z_HH = data.Z_HH[data.mask_HH];
	M_HH = data.M_HH[data.mask_HH];
	W_HH = data.W_HH[data.mask_HH];
	B_HH = data.B_HH[data.mask_HH];

	pttH_HH = pttH_HH[data.mask_ttH];
	Y_ttH = data.Y_ttH[data.mask_ttH];
	Z_ttH = data.Z_ttH[data.mask_ttH];
	M_ttH = data.M_ttH[data.mask_ttH];
	W_ttH = data.W_ttH[data.mask_ttH];
	B_ttH = data.B_ttH[data.mask_ttH]; 
 
	pZH_HH = pZH_HH[data.mask_ZH];
	Y_ZH = data.Y_ZH[data.mask_ZH];
	Z_ZH = data.Z_ZH[data.mask_ZH];
	M_ZH = data.M_ZH[data.mask_ZH];
	W_ZH = data.W_ZH[data.mask_ZH];
	B_ZH = data.B_ZH[data.mask_ZH];

	pJJ_HH = pJJ_HH[data.mask_JJ];
	Y_JJ = data.Y_JJ[data.mask_JJ];
	Z_JJ = data.Z_JJ[data.mask_JJ];
	M_JJ = data.M_JJ[data.mask_JJ];
	W_JJ = data.W_JJ[data.mask_JJ];
	B_JJ = data.B_JJ[data.mask_JJ];

	X = np.linspace(-20., 5., 500)
	Sig = []
	dSig = []
	XX = []
	
	Z_max = 0;
	dZ_max = 0;
	cut = 0;

	S = 0
	B = 0
	T = 0
	Zh = 0
	for x in X:
		
		sumW_sig_unc = 0
		sumW_bkg_unc = 0
		sumW_sig = 0
		sumW_bkg = 0
		t = 0
		z = 0
		sumW_bkg_cut = 0
		for i in range(0,pHH_HH.shape[0]):
			if pHH_HH[i] > x and Z_HH[i]*1e-3 < 130 and Z_HH[i]*1e-3 > 120 and B_HH[i]*1e-3 < 140 and B_HH[i]*1e-3 > 90:
				sumW_sig = sumW_sig+W_HH[i]*4
				sumW_sig_unc = sumW_sig_unc+(W_HH[i]*4*W_HH[i]*4)

		for i in range(0,pttH_HH.shape[0]):
			if pttH_HH[i] > x and Z_ttH[i]*1e-3 < 130 and Z_ttH[i]*1e-3 > 120 and B_ttH[i]*1e-3 < 140 and B_ttH[i]*1e-3 > 90:
				sumW_bkg = sumW_bkg+W_ttH[i]*4
				t += W_ttH[i]*4
				sumW_bkg_unc = sumW_bkg_unc+(W_ttH[i]*4*W_ttH[i]*4)
				

		for i in range(0,pZH_HH.shape[0]):
			if pZH_HH[i] > x and (Z_ZH[i]*1e-3 < 130 and Z_ZH[i]*1e-3 > 120) and (B_ZH[i]*1e-3 < 140 and B_ZH[i]*1e-3 > 90):
				sumW_bkg = sumW_bkg+W_ZH[i]*4
				z += W_ZH[i]*4
				sumW_bkg_unc = sumW_bkg_unc+(W_ZH[i]*4*W_ZH[i]*4)
					
			
		for i in range(0,pJJ_HH.shape[0]):
			if pJJ_HH[i] > x and (Z_JJ[i]*1e-3 < 130 and Z_JJ[i]*1e-3 > 120) and (B_JJ[i]*1e-3 < 140 and B_JJ[i]*1e-3 > 90):
				sumW_bkg = sumW_bkg+W_JJ[i]*4
				sumW_bkg_unc = sumW_bkg_unc+(W_JJ[i]*4*W_JJ[i]*4)
				if (Z_JJ[i]*1e-3 < 127 and Z_JJ[i]*1e-3 > 123):
					sumW_bkg_cut = sumW_bkg_cut + W_JJ[i]*4
				
				
		if sumW_bkg < 0.001 or sumW_bkg_cut < .8:
			break;

		sumW_sig_unc = math.sqrt(sumW_sig_unc)
		sumW_bkg_unc = math.sqrt(sumW_bkg_unc)
		Z = math.sqrt( 2*( (sumW_sig+sumW_bkg)*math.log(1+(sumW_sig/sumW_bkg)) - sumW_sig ) )
		if Z < 0.001:
			break;
		l=math.log(1+(sumW_sig/sumW_bkg));

		dZds = l/Z;
		dZdb = (l-sumW_sig/sumW_bkg)/Z;

		dZ=math.sqrt(dZds*dZds * sumW_sig_unc*sumW_sig_unc + dZdb*dZdb * sumW_bkg_unc*sumW_bkg_unc);
 
		Sig.append(Z)
		dSig.append(dZ)
		XX.append(x)

		print('X : %.3f , Z : %.3f , S : %.3f, B : %.3f' % (x,Z,sumW_sig,sumW_bkg))

		if Z > Z_max:
			Z_max = Z
			dZ_max = dZ
			cut = x
			S = sumW_sig
			B = sumW_bkg_cut
			T = t; Zh = z

	Z0 = np.ones(len(XX))*Sig[0];

	print('Z Maximum Z : %.3f +/- %.3f for category : %i  Cut : %.3f : Z0 : %.3f' %(Z_max, dZ_max, cat, cut, Sig[0]) )
	print(' S : %.3f , JJ : %.3f ' % (S,B))
	print(' ttH : %.3f , ZH : %.3f' % (T,Zh))
	return Z_max, Sig

def plot_Z_CutBased(HH,ttH,ZH,JJ,cat):

	Scale_HH  = [4., 4., 4., 4., 4.];  # SM
	#Scale_HH  = [3.984, 4.006, 3.912, 4.005, 3.993];  # +6
	#Scale_HH  = [4.014, 3.970, 4.081, 4.086, 4.008];  # +4
	#Scale_HH  = [4.119, 3.870, 4.018, 3.972, 3.984];  # +2
	#Scale_HH  = [3.876, 4.040, 3.991, 4.100, 4.038];  # -1
	#Scale_HH  = [4.124, 4.056, 3.989, 3.981, 4.007];  #  0
	#Scale_HH  = [4.065, 3.982, 3.891, 4.117, 4.031];  # -2
	#Scale_HH  = [4.010, 4.069, 3.937, 3.963, 3.998];  # -4
	#Scale_HH  = [3.849, 3.968, 4.110, 4.004, 3.980];  # -6
	Scale_ZH  = [4., 4., 4., 4., 4.];
	Scale_ttH = [4., 4., 4., 4., 4.];
	Scale_JJ  = [4., 4., 4., 4., 4.]; 

	pHH_HH  = HH[:,0]
	pttH_HH = ttH[:,0]
	pZH_HH  = ZH[:,0]
	pJJ_HH  = JJ[:,0]

	X = np.linspace(0., .99, 100)
	Sig = []
	dSig = []
	XX = []
	
	Z_max = 0;
	dZ_max = 0;
	cut = 0;
	for x in X:
		sumW_sig_unc = 0
		sumW_bkg_unc = 0
		sumW_jj_unc = 0
		sumW_sig = 0
		sumW_bkg = 0
		sumW_jj = 0
		for i in range(0,pHH_HH.shape[0]):
			if pHH_HH[i] > x and (data.Z_HH[i]*1e-3 < 130 and data.Z_HH[i]*1e-3 > 120):
				if cat == 5 and data.M_HH[i]*1e-3 > 350:
					sumW_sig = sumW_sig+data.W_HH[i]*Scale_HH[cat-1]
					sumW_sig_unc = sumW_sig_unc+(data.W_HH[i]*Scale_HH[cat-1]*data.W_HH[i]*Scale_HH[cat-1])
				elif data.C_HH[i] >= cat and data.M_HH[i]*1e-3 > 350:
					sumW_sig = sumW_sig+data.W_HH[i]*Scale_HH[cat-1]
					sumW_sig_unc = sumW_sig_unc+(data.W_HH[i]*Scale_HH[cat-1]*data.W_HH[i]*Scale_HH[cat-1])

		for i in range(0,pttH_HH.shape[0]):
			if pttH_HH[i] > x and (data.Z_ttH[i]*1e-3 < 130 and data.Z_ttH[i]*1e-3 > 120):
				if cat == 5 and data.M_ttH[i]*1e-3 > 350:
					sumW_bkg = sumW_bkg+data.W_ttH[i]*Scale_ttH[cat-1]
					sumW_bkg_unc = sumW_bkg_unc+(data.W_ttH[i]*Scale_ttH[cat-1]*data.W_ttH[i]*Scale_ttH[cat-1])
				elif data.C_ttH[i] >= cat and data.M_ttH[i]*1e-3 > 350:
					sumW_bkg = sumW_bkg+data.W_ttH[i]*Scale_ttH[cat-1]
					sumW_bkg_unc = sumW_bkg_unc+(data.W_ttH[i]*Scale_ttH[cat-1]*data.W_ttH[i]*Scale_ttH[cat-1])

		for i in range(0,pZH_HH.shape[0]):
			if pZH_HH[i] > x and (data.Z_ZH[i]*1e-3 < 130 and data.Z_ZH[i]*1e-3 > 120):
				if cat == 5 and data.M_ZH[i]*1e-3 > 350:
					sumW_bkg = sumW_bkg+data.W_ZH[i]*Scale_ZH[cat-1]
					sumW_bkg_unc = sumW_bkg_unc+(data.W_ZH[i]*Scale_ZH[cat-1]*data.W_ZH[i]*Scale_ZH[cat-1])
				elif data.C_ZH[i] >= cat and data.M_ZH[i]*1e-3 > 350:
					sumW_bkg = sumW_bkg+data.W_ZH[i]*Scale_ZH[cat-1]
					sumW_bkg_unc = sumW_bkg_unc+(data.W_ZH[i]*Scale_ZH[cat-1]*data.W_ZH[i]*Scale_ZH[cat-1])		
			
		for i in range(0,pJJ_HH.shape[0]):
			if pJJ_HH[i] > x and (data.Z_JJ[i]*1e-3 < 160 and data.Z_JJ[i]*1e-3 > 105):
				if cat == 5 and data.M_JJ[i]*1e-3 > 350:
					sumW_jj = sumW_jj+data.W_JJ[i]*Scale_JJ[cat-1]*10./55.
					sumW_jj_unc = sumW_jj_unc+(data.W_JJ[i]*Scale_JJ[cat-1]*10./55.*data.W_JJ[i]*Scale_JJ[cat-1]*10./55.)
				elif data.C_JJ[i] >= cat and data.M_JJ[i]*1e-3 > 350:
					sumW_jj = sumW_jj+data.W_JJ[i]*Scale_JJ[cat-1]*10./55.
					sumW_jj_unc = sumW_jj_unc+(data.W_JJ[i]*Scale_JJ[cat-1]*10./55.*data.W_JJ[i]*Scale_JJ[cat-1]*10./55.)
				
		if sumW_jj < 2.:
			continue;
		sumW_bkg_unc = sumW_bkg_unc+sumW_jj_unc
		sumW_bkg     = sumW_bkg+sumW_jj
		sumW_sig_unc = math.sqrt(sumW_sig_unc)
		sumW_bkg_unc = math.sqrt(sumW_bkg_unc)
		
		Z = math.sqrt( 2*( (sumW_sig+sumW_bkg)*math.log(1+(sumW_sig/sumW_bkg)) - sumW_sig))
		#print(Z,x)
		if Z < 0.001:
			continue;
		l=math.log(1+(sumW_sig/sumW_bkg));

		dZds = l/Z;
		dZdb = (l-sumW_sig/sumW_bkg)/Z;

		dZ=math.sqrt(dZds*dZds * sumW_sig_unc*sumW_sig_unc + dZdb*dZdb * sumW_bkg_unc*sumW_bkg_unc);
 
		Sig.append(Z)
		dSig.append(dZ)
		XX.append(x)

		if Z > Z_max:
			Z_max = Z
			dZ_max = dZ
			cut = x

	print('On CutBased : Z Maximum Z : %.3f +/- %.3f for category : %i  Cut : %.3f : Z0 : %.3f' %(Z_max, dZ_max, cat, cut, 0.36) )
	return Z_max, Sig

def compute_Loss(y_true, y_pred):

	return keras.losses.categorical_crossentropy(y_true, y_pred)
		
def main():
	"""
	The main function of train_DNN
	"""
	args = getArgs();
	noutput = int(args.noutput)
	dropout_rate = float(args.dropout_rate)
	learning_rate = float(args.learning_rate)
	nepochs = int(args.nepochs)
	batch_size = int(args.batch_size)
	n_neurons = int(args.nerouns)
	
	for i in range(0,1):
		history = train(nepochs,noutput,dropout_rate,learning_rate,batch_size,n_neurons)

		n_inputs = data.X_Train.shape[1];
		clf = Classifier('DNN_Classifier',n_inputs,noutput,dropout_rate,n_neurons).Sequential
		adm = optimizers.Adam(lr=learning_rate)
		clf.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['categorical_accuracy']);
		clf.load_weights('weights/best.weights')

		HH  = clf.predict(data.X_HH)
		ttH = clf.predict(data.X_ttH)
		ZH  = clf.predict(data.X_ZH)
		JJ  = clf.predict(data.X_JJ)

		y_pred = clf.predict(data.X_Test);

		cm = confusion_matrix(np.argmax(data.Y_Test, axis=1), np.argmax(y_pred, axis=1));


		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

		print(cm)

		pHH = Sim(HH[:,0], HH[:,1], HH[:,2], HH[:,3])
		pttH = Sim(ttH[:,0], ttH[:,1], ttH[:,2], ttH[:,3])
		pZH = Sim(ZH[:,0], ZH[:,1], ZH[:,2], ZH[:,3])
		pJJ = Sim(JJ[:,0], JJ[:,1], JJ[:,2], JJ[:,3])

		Z_max, Z = plot_Z(pHH,pttH,pZH,pJJ,5)
		
		dirname = "SM_model_Mbb_LO_2btag77"

		if not os.path.exists(dirname):
			os.mkdir(dirname)

		clf_json = clf.to_json()
		with open(dirname+"/"+dirname+".json", "w") as json_file:
    			json_file.write(clf_json)
		clf.save_weights(dirname+"/"+dirname+".h5")

		plot_confusion_matrix(cm,np.asarray(['HH','ttH','ZH','JJ']), normalize=True,title='Normalized confusion matrix',dirc=dirname)

		plt.figure()
		plt.hist(HH[:,0], 50, histtype='step',  density='True')
		plt.hist(ttH[:,0],50, histtype='step', density='True')
		plt.hist(ZH[:,0], 50, histtype='step', density='True')
		plt.hist(JJ[:,0], 50, histtype='step', density='True')
		plt.yscale('log')
		plt.xlabel('pHH')
		plt.ylabel('Event / 0.02')
		plt.legend(['HH','ttH','ZH','JJ'])
		plt.savefig(dirname+'/pHH.png')
		plt.close()


		plt.figure()

		plt.hist(pHH, 50, histtype='step',  density='True', weights= data.W_HH)
		plt.hist(pttH,50, histtype='step', density='True', weights= data.W_ttH)
		plt.hist(pZH, 50, histtype='step', density='True', weights= data.W_ZH)
		plt.hist(pJJ, 50, histtype='step', density='True', weights= data.W_JJ)
		plt.yscale('log')
		plt.xlabel('DHH')
		plt.ylabel('Event / 0.02')
		plt.legend(['HH','ttH','ZH','JJ'])
		plt.savefig(dirname+'/SigpHH.png')

		val_loss = np.asarray(history.history['val_loss']);
		train_loss = np.asarray(history.history['loss']);
		val_acc = np.asarray(history.history['val_categorical_accuracy']);
		train_acc = np.asarray(history.history['categorical_accuracy']);

		Y_Val_prob = clf.predict_proba(data.X_Val)
		plot_roc_curve(data.Y_Val, Y_Val_prob,dirname)
		
		plot_model_history(history,dirname)
		np.save(dirname+'/val_loss.npy',val_loss);
		np.save(dirname+'/train_loss.npy',train_loss);
		np.save(dirname+'/val_acc.npy',val_acc);
		np.save(dirname+'/train_acc.npy',train_acc);
		np.save(dirname+'/Z.npy',Z);

		exit()

		clf_json = clf.to_json()
		with open(dirname+"/SM_model.json", "w") as json_file:
    			json_file.write(clf_json)
		clf.save_weights(dirname+"/SM_model.h5")

		plt.figure()


		del clf;


def Sim(X,Y,Z,W):

	a = [math.log(X[i]*8.8111e-5/(Y[i]*0.00114975+Z[i]*0.00172452+W[i]*51.823)) for i in range(0,X.shape[0])]

	#a = [math.log(X[i]*0.000480766/(Y[i]*0.00114975+Z[i]*0.00172452+W[i]*51.823)) for i in range(0,X.shape[0])]

	return a

if __name__== '__main__':
	main()
	

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, roc_auc_score, auc
import keras
import tensorflow as tf
from scipy import interp

class Metrics(Callback):
	def on_train_begin(self, logs={}):
		self.val_f1s = []
		self.val_recalls = []
		self.val_precisions = []
 
	def on_epoch_end(self, epoch, logs={}):
		val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
		val_targ = self.validation_data[1]
		_val_f1 = f1_score(val_targ, val_predict, average='micro')
		_val_recall = recall_score(val_targ, val_predict, average='micro')
		_val_precision = precision_score(val_targ, val_predict, average='micro')
		self.val_f1s.append(_val_f1)
		self.val_recalls.append(_val_recall)
		self.val_precisions.append(_val_precision)
		print(' val_f1: %f val_precision: %f val_recall %f' % (_val_f1, _val_precision, _val_recall))
		return

class Roc(Callback):
	def on_train_begin(self, logs={}):
		self.HH_ROC = []
		self.JJ_ROC = []
		self.ZH_ROC = []
		self.ttH_ROC = []

	def on_train_end(self, log={}):
		val_predict = self.model.predict_proba(self.validation_data[0])
		val_targ = self.validation_data[1]
		roc_HH, roc_JJ, roc_ZH, roc_ttH = self.roc(val_targ, val_predict)
		self.HH_ROC.append(roc_HH)
		self.ZH_ROC.append(roc_ZH)
		self.JJ_ROC.append(roc_JJ)
		self.ttH_ROC.append(roc_ttH)
		print(' Final ')
		print(' ROC HH: %f ROC JJ: %f ROC ZH: %f ROC ttH: %f' % (roc_HH, roc_JJ, roc_ZH, roc_ttH))
		return
		

	def get_ROC(self):

		return self.HH_ROC, self.ttH_ROC, self.ZH_ROC, self.JJ_ROC

	def roc(self, Y_true, Y_flase, logs={}):
 
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		for i in range(4):

   			fpr[i], tpr[i], thresholds = roc_curve(Y_true[:, i], Y_flase[:, i])
   			roc_auc[i] = auc(fpr[i], tpr[i])

		fpr["micro"], tpr["micro"], thresholds = roc_curve(Y_true.ravel(), Y_flase.ravel())
		roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
		all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))
		mean_tpr = np.zeros_like(all_fpr)
		for i in range(4):
   			mean_tpr += interp(all_fpr, fpr[i], tpr[i])
		mean_tpr /= 4
		fpr["macro"] = all_fpr
		tpr["macro"] = mean_tpr
		roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
								
		return roc_auc[0], roc_auc[3], roc_auc[2], roc_auc[1]



def classifier_loss(y_true, y_pred, w):
    return tf.losses.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred, sample_weight=w)


def oneplus_cosine_proximity(y_true, y_pred):
    return 1 + keras.losses.cosine_proximity(y_true, y_pred)
 


#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
from sklearn import preprocessing

class Data:
	"""
	Data class for data pre-processing
	Training type DNN or ANN
	"""

	def __init__(self,Kappa_lambda, Type='None'):

		self.Type = Type;
		self.Lambda = Kappa_lambda;
		self.path = '/afs/cern.ch/work/m/mobelfki/ANN/Array/h025_77btag2/';
		self.loadGlobalData();
		self.loadTestData();
		self.loadValData();
		self.loadTrainData();
		self.loadTestData();
		self.loadValData();
		self.loadForProcessing();
		self.setXYZ();
		#self.ProcessData();
		self.ApplyMassCut();

	def setXYZ(self):
		
		c = -9

		self.X = self.GlobalData[:,:c];
		self.Y = self.GlobalData[:,-4:];
		self.Z = self.GlobalData[:,-9];
		self.B = self.GlobalData[:,-10];
		self.W = self.GlobalData[:,-8];
		self.MC = self.GlobalData[:,-6];
		
		self.X_Train = self.TrainData[:,:c];
		self.Y_Train = self.TrainData[:,-4:];
		self.Z_Train = self.TrainData[:,-9];
		self.B_Train = self.TrainData[:,-10];
		self.W_Train = self.TrainData[:,-8];
		self.MC_Train = self.TrainData[:,-6];

		self.M_Train = self.TrainData[:,-5];
	
		self.X_Val = self.ValData[:,:c];
		self.Y_Val = self.ValData[:,-4:];
		self.Z_Val = self.ValData[:,-9];
		self.B_Val = self.ValData[:,-10];
		self.W_Val = self.ValData[:,-8];
		self.MC_Val = self.ValData[:,-6];

		self.M_Val = self.ValData[:,-5];

		self.X_Test = self.TestData[:,:c];
		self.Y_Test = self.TestData[:,-4:];
		self.Z_Test = self.TestData[:,-9];
		self.B_Test = self.TestData[:,-10];
		self.W_Test = self.TestData[:,-8];
		self.MC_Test = self.TestData[:,-6];

		self.M_Test = self.TestData[:,-5];

		self.X_HH = self.HH[:,:c];
		self.Y_HH = self.HH[:,-4:];
		self.Z_HH = self.HH[:,-9];
		self.B_HH = self.HH[:,-10];
		self.M_HH = self.HH[:,-5];
		self.W_HH = self.HH[:,-8];
		self.C_HH = self.HH[:,-7];

		self.X_ttH = self.ttH[:,:c];
		self.Y_ttH = self.ttH[:,-4:];
		self.Z_ttH = self.ttH[:,-9];
		self.B_ttH = self.ttH[:,-10];
		self.M_ttH = self.ttH[:,-5];
		self.W_ttH = self.ttH[:,-8];
		self.C_ttH = self.ttH[:,-7]; 
 
		self.X_ZH = self.ZH[:,:c];
		self.Y_ZH = self.ZH[:,-4:];
		self.Z_ZH = self.ZH[:,-9];
		self.B_ZH = self.ZH[:,-10];
		self.M_ZH = self.ZH[:,-5];
		self.W_ZH = self.ZH[:,-8];
		self.C_ZH = self.ZH[:,-7];

		self.X_H7_ZH = self.H7_ZH[:,:c];
		self.Y_H7_ZH = self.H7_ZH[:,-4:];
		self.Z_H7_ZH = self.H7_ZH[:,-9];
		self.B_H7_ZH = self.H7_ZH[:,-10];
		self.M_H7_ZH = self.H7_ZH[:,-5];
		self.W_H7_ZH = self.H7_ZH[:,-8];
		self.C_H7_ZH = self.H7_ZH[:,-7];

		self.X_H7_ttH = self.H7_ttH[:,:c];
		self.Y_H7_ttH = self.H7_ttH[:,-4:];
		self.Z_H7_ttH = self.H7_ttH[:,-9];
		self.B_H7_ttH = self.H7_ttH[:,-10];
		self.M_H7_ttH = self.H7_ttH[:,-5];
		self.W_H7_ttH = self.H7_ttH[:,-8];
		self.C_H7_ttH = self.H7_ttH[:,-7]; 

		self.X_JJ = self.JJ[:,:c];
		self.Y_JJ = self.JJ[:,-4:];
		self.Z_JJ = self.JJ[:,-9];
		self.B_JJ = self.JJ[:,-10];
		self.M_JJ = self.JJ[:,-5];
		self.W_JJ = self.JJ[:,-8];
		self.C_JJ = self.JJ[:,-7];

		self.X_Data = self.RealData[:,:c];
		self.Y_Data = self.RealData[:,-4:];
		self.Z_Data = self.RealData[:,-9];
		self.B_Data = self.RealData[:,-10];
		self.W_Data = self.RealData[:,-8];
		self.M_Data = self.RealData[:,-5];
		self.C_Data = self.RealData[:,-7];


		self.Process = self.Process[:,:c];

	def loadGlobalData(self):
		hh = "local";
		if self.Lambda == 1 :
			if self.Type == 'LO':
				hh   = np.load(self.path+'output.aMCnloHwpp_hh_yybb_AF2.MxAODDetailedNoSkim_h025.root.Tree.npy');
			if self.Type == 'NLO':
				hh   = np.load(self.path+'output.PowhegH7_HHbbyy_cHHH01d0.MxAODDetailedNoSkim_h025.root.Tree.npy');
		if self.Lambda == 10 :
			if self.Type == 'LO':
				hh   = np.load(self.path+'output.MGPy8_hh_yybb_plus_lambda10_AF2.MxAODDetailedNoSkim_h025.root.Tree.npy');
			if self.Type == 'NLO':
				hh   = np.load(self.path+'output.PowhegH7_HHbbyy_cHHH10d0.MxAODDetailedNoSkim_h025.root.Tree.npy');
		elif self.Lambda == 6:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_plus_lambda06_AF2.MxAODDetailedNoSkim_h025.root.Tree.npy');
		elif self.Lambda == 4:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_plus_lambda04_AF2.MxAODDetailedNoSkim_h025.root.Tree.npy');
		elif self.Lambda == 2:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_plus_lambda02_AF2.MxAODDetailedNoSkim_h025.root.Tree.npy');
		elif self.Lambda == 0:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_lambda00_AF2.MxAODDetailedNoSkim_h025.root.Tree.npy');
		elif self.Lambda == -1:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_minus_lambda01_AF2.MxAODDetailedNoSkim_h025.root.Tree.npy');
		elif self.Lambda == -2:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_minus_lambda02_AF2.MxAODDetailedNoSkim_h025.root.Tree.npy');
		elif self.Lambda == -4:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_minus_lambda04_AF2.MxAODDetailedNoSkim_h025.root.Tree.npy');
		elif self.Lambda == -6:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_minus_lambda06_AF2.MxAODDetailedNoSkim_h025.root.Tree.npy');

		tth  = np.load(self.path+'output.PowhegPy8_ttH125_fixweight.MxAODDetailed_h025.root.Tree.npy');
		zh   = np.load(self.path+'output.PowhegPy8_ZH125J.MxAODDetailed_h025.root.Tree.npy');
		jj   = np.load(self.path+'output.Sherpa2_diphoton_myy_90_175.MxAODDetailed_h025.root.Tree.npy');
		twh  = np.load(self.path+'output.aMCnloPy8_tWH125.MxAODDetailed_h025.root.Tree.npy');
		ggzh = np.load(self.path+'output.PowhegPy8_ggZH125.MxAODDetailed_h025.root.Tree.npy');
		bbh  = np.load(self.path+'output.PowhegPy8_bbH125.MxAODDetailed_h025.root.Tree.npy');
		ggh  = np.load(self.path+'output.PowhegPy8_NNLOPS_ggH125.MxAODDetailed_h025.root.Tree.npy');
	
		tth = np.append(tth, twh, axis=0);
		zh = np.append(ggzh, zh,  axis=0);
		jj = np.append(jj, bbh,  axis=0);
		jj = np.append(jj, ggh,  axis=0);

		hh = np.append(hh, tth, axis=0);
		hh = np.append(hh, zh, axis=0);
		hh = np.append(hh, jj, axis=0);

		self.RealData = np.load(self.path+'output.DataFullRun2.root.Tree.npy')

		#self.GlobalData = self.Suffle(hh,tth,zh,jj);

		self.GlobalData = hh;

	def loadTrainData(self):
		hh = "local";
		if self.Lambda == 1 :
			if self.Type == 'LO':
				hh   = np.load(self.path+'output.aMCnloHwpp_hh_yybb_AF2.MxAODDetailedNoSkim_h025.root.Tree_Train.npy');
			if self.Type == 'NLO':
				hh   = np.load(self.path+'output.PowhegH7_HHbbyy_cHHH01d0.MxAODDetailedNoSkim_h025.root.Tree_Train.npy');
		if self.Lambda == 10 :
			if self.Type == 'LO':
				hh   = np.load(self.path+'output.MGPy8_hh_yybb_plus_lambda10_AF2.MxAODDetailedNoSkim_h025.root.Tree_Train.npy');
			if self.Type == 'NLO':
				hh   = np.load(self.path+'output.PowhegH7_HHbbyy_cHHH10d0.MxAODDetailedNoSkim_h025.root.Tree_Train.npy');
		elif self.Lambda == 6:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_plus_lambda06_AF2.MxAODDetailedNoSkim_h025.root.Tree_Train.npy');
		elif self.Lambda == 4:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_plus_lambda04_AF2.MxAODDetailedNoSkim_h025.root.Tree_Train.npy');
		elif self.Lambda == 2:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_plus_lambda02_AF2.MxAODDetailedNoSkim_h025.root.Tree_Train.npy');
		elif self.Lambda == 0:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_lambda00_AF2.MxAODDetailedNoSkim_h025.root.Tree_Train.npy');
		elif self.Lambda == -1:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_minus_lambda01_AF2.MxAODDetailedNoSkim_h025.root.Tree_Train.npy');
		elif self.Lambda == -2:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_minus_lambda02_AF2.MxAODDetailedNoSkim_h025.root.Tree_Train.npy');
		elif self.Lambda == -4:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_minus_lambda04_AF2.MxAODDetailedNoSkim_h025.root.Tree_Train.npy');
		elif self.Lambda == -6:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_minus_lambda06_AF2.MxAODDetailedNoSkim_h025.root.Tree_Train.npy');

		tth  = np.load(self.path+'output.PowhegPy8_ttH125_fixweight.MxAODDetailed_h025.root.Tree_Train.npy');
		zh   = np.load(self.path+'output.PowhegPy8_ZH125J.MxAODDetailed_h025.root.Tree_Train.npy');
		jj   = np.load(self.path+'output.Sherpa2_diphoton_myy_90_175.MxAODDetailed_h025.root.Tree_Train.npy');
		twh  = np.load(self.path+'output.aMCnloPy8_tWH125.MxAODDetailed_h025.root.Tree_Train.npy');
		ggzh = np.load(self.path+'output.PowhegPy8_ggZH125.MxAODDetailed_h025.root.Tree_Train.npy');
		bbh  = np.load(self.path+'output.PowhegPy8_bbH125.MxAODDetailed_h025.root.Tree_Train.npy');
		ggh  = np.load(self.path+'output.PowhegPy8_NNLOPS_ggH125.MxAODDetailed_h025.root.Tree_Train.npy');

		tth = np.append(tth, twh, axis=0);
		zh = np.append(ggzh, zh,  axis=0);
		jj = np.append(jj, bbh,  axis=0);
		jj = np.append(jj, ggh,  axis=0);

		hh = np.append(hh, tth, axis=0);
		hh = np.append(hh, zh, axis=0);
		hh = np.append(hh, jj, axis=0);

		#self.TrainData = self.Suffle(hh,tth,zh,jj);

		hh = np.append(hh, self.ValData, axis=0);
		
		self.TrainData = hh;

	def loadForProcessing(self):
	
		hh   = np.load(self.path+'output.aMCnloHwpp_hh_yybb_AF2.MxAODDetailedNoSkim_h025.root.Tree_Train.npy');
		tth  = np.load(self.path+'output.PowhegPy8_ttH125_fixweight.MxAODDetailed_h025.root.Tree_Train.npy');
		zh   = np.load(self.path+'output.PowhegPy8_ZH125J.MxAODDetailed_h025.root.Tree_Train.npy');
		jj   = np.load(self.path+'output.Sherpa2_diphoton_myy_90_175.MxAODDetailed_h025.root.Tree_Train.npy');
		twh  = np.load(self.path+'output.aMCnloPy8_tWH125.MxAODDetailed_h025.root.Tree_Train.npy');
		ggzh = np.load(self.path+'output.PowhegPy8_ggZH125.MxAODDetailed_h025.root.Tree_Train.npy');
		bbh  = np.load(self.path+'output.PowhegPy8_bbH125.MxAODDetailed_h025.root.Tree_Train.npy');
		ggh  = np.load(self.path+'output.PowhegPy8_NNLOPS_ggH125.MxAODDetailed_h025.root.Tree_Train.npy');

		tth = np.append(tth, twh, axis=0);
		zh = np.append(ggzh, zh,  axis=0);
		jj = np.append(jj, bbh,  axis=0);
		jj = np.append(jj, ggh,  axis=0);
		
		hh = np.append(hh, tth, axis=0);
		hh = np.append(hh, zh, axis=0);
		hh = np.append(hh, jj, axis=0);

		self.Process = hh;

	def loadTestData(self):
		hh = "local";
		if self.Lambda == 1 :
			if self.Type == 'LO':
				hh   = np.load(self.path+'output.aMCnloHwpp_hh_yybb_AF2.MxAODDetailedNoSkim_h025.root.Tree_Test.npy');
			if self.Type == 'NLO':
				hh   = np.load(self.path+'output.PowhegH7_HHbbyy_cHHH01d0.MxAODDetailedNoSkim_h025.root.Tree_Test.npy');
		if self.Lambda == 10 :
			if self.Type == 'LO':
				hh   = np.load(self.path+'output.MGPy8_hh_yybb_plus_lambda10_AF2.MxAODDetailedNoSkim_h025.root.Tree_Test.npy');
			if self.Type == 'NLO':
				hh   = np.load(self.path+'output.PowhegH7_HHbbyy_cHHH10d0.MxAODDetailedNoSkim_h025.root.Tree_Test.npy');
		elif self.Lambda == 6:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_plus_lambda06_AF2.MxAODDetailedNoSkim_h025.root.Tree_Test.npy');
		elif self.Lambda == 4:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_plus_lambda04_AF2.MxAODDetailedNoSkim_h025.root.Tree_Test.npy');
		elif self.Lambda == 2:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_plus_lambda02_AF2.MxAODDetailedNoSkim_h025.root.Tree_Test.npy');
		elif self.Lambda == 0:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_lambda00_AF2.MxAODDetailedNoSkim_h025.root.Tree_Test.npy');
		elif self.Lambda == -1:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_minus_lambda01_AF2.MxAODDetailedNoSkim_h025.root.Tree_Test.npy');
		elif self.Lambda == -2:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_minus_lambda02_AF2.MxAODDetailedNoSkim_h025.root.Tree_Test.npy');
		elif self.Lambda == -4:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_minus_lambda04_AF2.MxAODDetailedNoSkim_h025.root.Tree_Test.npy');
		elif self.Lambda == -6:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_minus_lambda06_AF2.MxAODDetailedNoSkim_h025.root.Tree_Test.npy');

		ggzh = np.load(self.path+'output.PowhegPy8_ggZH125.MxAODDetailed_h025.root.Tree_Test.npy');
		twh  = np.load(self.path+'output.aMCnloPy8_tWH125.MxAODDetailed_h025.root.Tree_Test.npy');
		tth  = np.load(self.path+'output.PowhegPy8_ttH125_fixweight.MxAODDetailed_h025.root.Tree_Test.npy');
		zh   = np.load(self.path+'output.PowhegPy8_ZH125J.MxAODDetailed_h025.root.Tree_Test.npy');
		jj   = np.load(self.path+'output.Sherpa2_diphoton_myy_90_175.MxAODDetailed_h025.root.Tree_Test.npy');
		bbh  = np.load(self.path+'output.PowhegPy8_bbH125.MxAODDetailed_h025.root.Tree_Test.npy');
		ggh  = np.load(self.path+'output.PowhegPy8_NNLOPS_ggH125.MxAODDetailed_h025.root.Tree_Test.npy');

		h7_zh   = np.load(self.path+'output.PowhegH713_ZH125J.MxAODDetailed_h025.root.Tree_Test.npy');
		h7_ggzh = np.load(self.path+'output.PowhegH713_ZH125J.MxAODDetailed_h025.root.Tree_Test.npy');
		h7_tth  = np.load(self.path+'output.PowhegHw7_ttH125_fixweight.MxAODDetailed_h025.root.Tree_Test.npy');

		tth = np.append(tth, twh, axis=0);
		zh = np.append(ggzh, zh,  axis=0);
		h7_zh = np.append(h7_ggzh, h7_zh,  axis=0);
		tth = np.append(tth, bbh,  axis=0);
		tth = np.append(tth, ggh,  axis=0);
		
		self.HH  = hh
		self.ttH = tth
		self.ZH  = zh
		self.H7_ZH = h7_zh
		self.H7_ttH = h7_tth
		self.JJ  = jj

		hh = np.append(hh, tth, axis=0);
		hh = np.append(hh, zh, axis=0);
		hh = np.append(hh, jj, axis=0);

		self.TestData = hh;

	def loadValData(self):
		hh = "local";
		if self.Lambda == 1 :
			if self.Type == 'LO':
				hh   = np.load(self.path+'output.aMCnloHwpp_hh_yybb_AF2.MxAODDetailedNoSkim_h025.root.Tree_Val.npy');
			if self.Type == 'NLO':
				hh   = np.load(self.path+'output.PowhegH7_HHbbyy_cHHH01d0.MxAODDetailedNoSkim_h025.root.Tree_Val.npy');
		if self.Lambda == 10 :
			if self.Type == 'LO':
				hh   = np.load(self.path+'output.MGPy8_hh_yybb_plus_lambda10_AF2.MxAODDetailedNoSkim_h025.root.Tree_Val.npy');
			if self.Type == 'NLO':
				hh   = np.load(self.path+'output.PowhegH7_HHbbyy_cHHH10d0.MxAODDetailedNoSkim_h025.root.Tree_Val.npy');
		elif self.Lambda == 6:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_plus_lambda06_AF2.MxAODDetailedNoSkim_h025.root.Tree_Val.npy');
		elif self.Lambda == 4:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_plus_lambda04_AF2.MxAODDetailedNoSkim_h025.root.Tree_Val.npy');
		elif self.Lambda == 2:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_plus_lambda02_AF2.MxAODDetailedNoSkim_h025.root.Tree_Val.npy');
		elif self.Lambda == 0:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_lambda00_AF2.MxAODDetailedNoSkim_h025.root.Tree_Val.npy');
		elif self.Lambda == -1:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_minus_lambda01_AF2.MxAODDetailedNoSkim_h025.root.Tree_Val.npy');
		elif self.Lambda == -2:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_minus_lambda02_AF2.MxAODDetailedNoSkim_h025.root.Tree_Val.npy');
		elif self.Lambda == -4:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_minus_lambda04_AF2.MxAODDetailedNoSkim_h025.root.Tree_Val.npy');
		elif self.Lambda == -6:
			hh   = np.load(self.path+'output.MGPy8_hh_yybb_minus_lambda06_AF2.MxAODDetailedNoSkim_h025.root.Tree_Val.npy');

		tth  = np.load(self.path+'output.PowhegPy8_ttH125_fixweight.MxAODDetailed_h025.root.Tree_Val.npy');
		zh   = np.load(self.path+'output.PowhegPy8_ZH125J.MxAODDetailed_h025.root.Tree_Val.npy');
		jj   = np.load(self.path+'output.Sherpa2_diphoton_myy_90_175.MxAODDetailed_h025.root.Tree_Val.npy');
		twh  = np.load(self.path+'output.aMCnloPy8_tWH125.MxAODDetailed_h025.root.Tree_Val.npy');
		ggzh = np.load(self.path+'output.PowhegPy8_ggZH125.MxAODDetailed_h025.root.Tree_Val.npy');
		bbh  = np.load(self.path+'output.PowhegPy8_bbH125.MxAODDetailed_h025.root.Tree_Val.npy');
		ggh  = np.load(self.path+'output.PowhegPy8_NNLOPS_ggH125.MxAODDetailed_h025.root.Tree_Val.npy');

		tth = np.append(tth, twh, axis=0);
		zh = np.append(ggzh, zh,  axis=0);
		jj = np.append(jj, bbh,  axis=0);
		jj = np.append(jj, ggh,  axis=0);

		hh = np.append(hh, tth, axis=0);
		hh = np.append(hh, zh, axis=0);
		hh = np.append(hh, jj, axis=0);

		#self.ValData = self.Suffle(hh,tth,zh,jj);

		self.ValData = hh;

	def ProcessData(self):

		"""
		min_max_scaler = preprocessing.MaxAbsScaler().fit(self.X_Train)

		self.X_Train = min_max_scaler.transform(self.X_Train)
		self.X_Val   = min_max_scaler.transform(self.X_Val)
		self.X_Test  = min_max_scaler.transform(self.X_Test)
		self.X_HH    = min_max_scaler.transform(self.X_HH)
		self.X_ZH    = min_max_scaler.transform(self.X_ZH)
		self.X_ttH   = min_max_scaler.transform(self.X_ttH)
		self.X_JJ    = min_max_scaler.transform(self.X_JJ)
		
		"""
		scaler = preprocessing.StandardScaler().fit(self.Process)

		self.X_Train = scaler.transform(self.X_Train)
		self.X_Val   = scaler.transform(self.X_Val)
		self.X_Test  = scaler.transform(self.X_Test)
		self.X_HH    = scaler.transform(self.X_HH)
		self.X_ZH    = scaler.transform(self.X_ZH)
		self.X_ttH   = scaler.transform(self.X_ttH)
		self.X_JJ    = scaler.transform(self.X_JJ)
		
		

	def SetClassWieght(self, X):

		
		#sumW = 0;
		#for row in X:
		#	if row[-4] == 1:
		#		sumW = row[-5]
		#		break
		for row in X:
			if row[-4] == 1:
				row[-6] = row[-6] * 1.
			if row[-3] == 1:
				row[-6] = row[-6] * 0.089
			if row[-2] == 1:
  				row[-6] = row[-6] * 4.913
			if row[-1] == 1:
				row[-6] = row[-6] * 0.8921			
	
		
	def Rescale(self, X):
		
		X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
		X = X_std * (1 - 0) + 0  # (max - min) + min [min,max]

	def Suffle(self, X, Y, Z, W):

		i = 0
		out = np.empty((0,X.shape[1]))
		for row in Z:
			reslt = np.vstack((row,Y[i],Z[i],W[i]))
			out = np.append(out, reslt,axis=0)
			i = i+1

		return out

	def ApplyMassCut(self):
	
		mhh = 350
		
		self.mask_HH = self.M_HH * 1e-3 > mhh
		self.mask_ZH = self.M_ZH * 1e-3 > mhh
		self.mask_ttH = self.M_ttH * 1e-3 > mhh
		self.mask_JJ = self.M_JJ * 1e-3 > mhh

		self.mask_H7_ZH = self.M_H7_ZH * 1e-3 > mhh

		self.mask_Train = self.M_Train * 1e-3 > mhh
		self.mask_Val   = self.M_Val * 1e-3 > mhh
		self.mask_Test  = self.M_Test * 1e-3 > mhh
	
		#self.mask_Train = self.mask_Train * (self.B_Train * 1e-3 < 140) * (self.B_Train * 1e-3 > 90)
		#self.mask_Test = self.mask_Test * (self.B_Test * 1e-3 < 140) * (self.B_Test * 1e-3 > 90)
		#self.mask_Val = self.mask_Val * (self.B_Val * 1e-3 < 140) * (self.B_Val * 1e-3 > 90)
		
		self.X_Train = self.X_Train[self.mask_Train];
		self.Y_Train = self.Y_Train[self.mask_Train];
		self.Z_Train = self.Z_Train[self.mask_Train];
		self.B_Train = self.B_Train[self.mask_Train];
		self.W_Train = self.W_Train[self.mask_Train];
		self.M_Train = self.M_Train[self.mask_Train];
		self.MC_Train = self.MC_Train[self.mask_Train];

		self.X_Val = self.X_Val[self.mask_Val];
		self.Y_Val = self.Y_Val[self.mask_Val];
		self.Z_Val = self.Z_Val[self.mask_Val];
		self.B_Val = self.B_Val[self.mask_Val];
		self.W_Val = self.W_Val[self.mask_Val];
		self.M_Val = self.M_Val[self.mask_Val];
		self.MC_Val = self.MC_Val[self.mask_Val];
	
		self.X_Test = self.X_Test[self.mask_Test];
		self.Y_Test = self.Y_Test[self.mask_Test];
		self.Z_Test = self.Z_Test[self.mask_Test];
		self.B_Test = self.B_Test[self.mask_Test];
		self.W_Test = self.W_Test[self.mask_Test];
		self.M_Test = self.M_Test[self.mask_Test];
		
		"""
		self.X_HH = self.X_HH[mask_HH];
		self.Y_HH = self.Y_HH[mask_HH];
		self.Z_HH = self.Z_HH[mask_HH];
		self.M_HH = self.M_HH[mask_HH];
		self.W_HH = self.W_HH[mask_HH];
		self.C_HH = self.C_HH[mask_HH];

		self.X_ttH = self.X_ttH[mask_ttH];
		self.Y_ttH = self.Y_ttH[mask_ttH];
		self.Z_ttH = self.Z_ttH[mask_ttH];
		self.M_ttH = self.M_ttH[mask_ttH];
		self.W_ttH = self.W_ttH[mask_ttH];
		self.C_ttH = self.C_ttH[mask_ttH]; 
 
		self.X_ZH = self.X_ZH[mask_ZH];
		self.Y_ZH = self.Y_ZH[mask_ZH];
		self.Z_ZH = self.Z_ZH[mask_ZH];
		self.M_ZH = self.M_ZH[mask_ZH];
		self.W_ZH = self.W_ZH[mask_ZH];
		self.C_ZH = self.C_ZH[mask_ZH];

		self.X_JJ = self.X_JJ[mask_JJ];
		self.Y_JJ = self.Y_JJ[mask_JJ];
		self.Z_JJ = self.Z_JJ[mask_JJ];
		self.M_JJ = self.M_JJ[mask_JJ];
		self.W_JJ = self.W_JJ[mask_JJ];
		self.C_JJ = self.C_JJ[mask_JJ];
		"""

	def getData(self, t):
	
		if t == 'X':
			return self.X;
		if t == 'Y':
			return self.Y;
		if t == 'Z':
			return self.Z;
		if t == 'HH':
			return self.X_HH;
		if t == 'ttH':
			return self.X_ttH;
		if t == 'ZH':
			return self.X_ZH;
		if t == 'JJ':
			return self.X_JJ;
		
	
		

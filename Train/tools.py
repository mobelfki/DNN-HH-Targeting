#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataProcessing import Data
import os
import math


data = Data(1,'NLO');

mhh = 350.

alpha = 4.

mbb_min = 90
mbb_max = 140

def Apply_Z_Cut_High(HH,ttH,ZH,JJ,cat,cut):

	Scale_HH  = [3.117, 4.090, 4.294, 4.303, alpha];  # SM
	#Scale_HH  = [3.984, 4.006, 3.912, 4.005, 3.993];  # +6
	#Scale_HH  = [4.014, 3.970, 4.081, 4.086, 4.008];  # +4
	#Scale_HH  = [4.119, 3.870, 4.018, 3.972, 3.984];  # +2
	#Scale_HH  = [3.876, 4.040, 3.991, 4.100, 4.038];  # -1
	#Scale_HH  = [4.124, 4.056, 3.989, 3.981, 4.007];  #  0
	#Scale_HH  = [4.065, 3.982, 3.891, 4.117, 4.031];  # -2
	#Scale_HH  = [4.010, 4.069, 3.937, 3.963, 3.998];  # -4
	#Scale_HH  = [3.849, 3.968, 4.110, 4.004, 3.980];  # -6
	Scale_ZH  = [4.177, 4.170, 4.270, 3.957, alpha];
	Scale_ttH = [4.042, 4.028, 3.970, 4.010, alpha];
	Scale_JJ  = [3.874, 4.245, 5.172, 3.397, alpha]; 

	#pHH_HH  = HH[:,0]
	#pttH_HH = ttH[:,0]
	#pZH_HH  = ZH[:,0]
	#pJJ_HH  = JJ[:,0]

	pHH_HH  = np.array(HH)
	pttH_HH = np.array(ttH)
	pZH_HH  = np.array(ZH)
	pJJ_HH  = np.array(JJ)

	x1 = cut[0]
	x2 = cut[1]
	HH_yields  = []
	ttH_yields = []
	ZH_yields  = []
	JJ_yields  = []

	mbb = []
	myy = []

	mask_HH = data.M_HH * 1e-3 > mhh
	mask_ZH = data.M_ZH * 1e-3 > mhh
	mask_ttH = data.M_ttH * 1e-3 > mhh
	mask_JJ = data.M_JJ * 1e-3 > mhh

	#pHH_HH = pHH_HH[mask_HH];
	Y_HH = data.Y_HH[mask_HH];
	Z_HH = data.Z_HH[mask_HH];
	M_HH = data.M_HH[mask_HH];
	W_HH = data.W_HH[mask_HH];
	B_HH = data.B_HH[mask_HH];

	#pttH_HH = pttH_HH[mask_ttH];
	Y_ttH = data.Y_ttH[mask_ttH];
	Z_ttH = data.Z_ttH[mask_ttH];
	M_ttH = data.M_ttH[mask_ttH];
	W_ttH = data.W_ttH[mask_ttH];
	B_ttH = data.B_ttH[mask_ttH]; 
 
	#pZH_HH = pZH_HH[mask_ZH];
	Y_ZH = data.Y_ZH[mask_ZH];
	Z_ZH = data.Z_ZH[mask_ZH];
	M_ZH = data.M_ZH[mask_ZH];
	W_ZH = data.W_ZH[mask_ZH];
	B_ZH = data.B_ZH[mask_ZH];

	#pJJ_HH = pJJ_HH[mask_JJ];
	Y_JJ = data.Y_JJ[mask_JJ];
	Z_JJ = data.Z_JJ[mask_JJ];
	M_JJ = data.M_JJ[mask_JJ];
	W_JJ = data.W_JJ[mask_JJ];
	B_JJ = data.B_JJ[mask_JJ];

	if (pHH_HH.shape[0]!=Z_HH.shape[0]):
		print('Problem')

	for i in range(0,pHH_HH.shape[0]):
		if pHH_HH[i] > x1 and pHH_HH[i] < x2 and (Z_HH[i]*1e-3 < 130 and Z_HH[i]*1e-3 > 120) and (B_HH[i]*1e-3 < mbb_max and B_HH[i]*1e-3 > mbb_min):
			if cat == 5 and M_HH[i]*1e-3 > mhh:
				HH_yields.append(W_HH[i]*alpha)
				mbb.append(B_HH[i]*1e-3)
				myy.append(Z_HH[i]*1e-3)

	for i in range(0,pttH_HH.shape[0]):
		if pttH_HH[i] > x1 and pttH_HH[i] < x2 and (Z_ttH[i]*1e-3 < 130 and Z_ttH[i]*1e-3 > 120) and (B_ttH[i]*1e-3 < mbb_max and B_ttH[i]*1e-3 > mbb_min):
			if cat == 5 and M_ttH[i]*1e-3 > mhh:
				ttH_yields.append(W_ttH[i]*alpha)
			

	for i in range(0,pZH_HH.shape[0]):
		if pZH_HH[i] > x1 and pZH_HH[i] < x2 and (Z_ZH[i]*1e-3 < 130 and Z_ZH[i]*1e-3 > 120) and (B_ZH[i]*1e-3 < mbb_max and B_ZH[i]*1e-3 > mbb_min):
			if cat == 5 and M_ZH[i]*1e-3 > mhh:
				ZH_yields.append(W_ZH[i]*alpha)
			
			
	for i in range(0,pJJ_HH.shape[0]):
		if pJJ_HH[i] > x1 and pJJ_HH[i] < x2 and (Z_JJ[i]*1e-3 < 130 and Z_JJ[i]*1e-3 > 120)and (B_JJ[i]*1e-3 < mbb_max and B_JJ[i]*1e-3 > mbb_min):
			if cat == 5 and M_JJ[i]*1e-3 > mhh:
				JJ_yields.append(W_JJ[i]*alpha)

	return np.asarray(HH_yields), np.asarray(ttH_yields), np.asarray(ZH_yields), np.asarray(JJ_yields), np.asarray(mbb), np.asarray(myy)


def Apply_Z_Cut_Low(HH,ttH,ZH,JJ,cat,cut):

	Scale_HH  = [3.117, 4.090, 4.294, 4.303, alpha];  # SM
	#Scale_HH  = [3.984, 4.006, 3.912, 4.005, 3.993];  # +6
	#Scale_HH  = [4.014, 3.970, 4.081, 4.086, 4.008];  # +4
	#Scale_HH  = [4.119, 3.870, 4.018, 3.972, 3.984];  # +2
	#Scale_HH  = [3.876, 4.040, 3.991, 4.100, 4.038];  # -1
	#Scale_HH  = [4.124, 4.056, 3.989, 3.981, 4.007];  #  0
	#Scale_HH  = [4.065, 3.982, 3.891, 4.117, 4.031];  # -2
	#Scale_HH  = [4.010, 4.069, 3.937, 3.963, 3.998];  # -4
	#Scale_HH  = [3.849, 3.968, 4.110, 4.004, 3.980];  # -6
	Scale_ZH  = [4.177, 4.170, 4.270, 3.957, alpha];
	Scale_ttH = [4.042, 4.028, 3.970, 4.010, alpha];
	Scale_JJ  = [3.874, 4.245, 5.172, 3.397, alpha]; 

	#pHH_HH  = HH[:,0]
	#pttH_HH = ttH[:,0]
	#pZH_HH  = ZH[:,0]
	#pJJ_HH  = JJ[:,0]

	pHH_HH  = np.array(HH)
	pttH_HH = np.array(ttH)
	pZH_HH  = np.array(ZH)
	pJJ_HH  = np.array(JJ)

	x1 = cut[0]
	x2 = cut[1]
	HH_yields  = []
	ttH_yields = []
	ZH_yields  = []
	JJ_yields  = []

	mbb = []
	myy = []

	mask_HH = data.M_HH * 1e-3 < mhh
	mask_ZH = data.M_ZH * 1e-3 < mhh
	mask_ttH = data.M_ttH * 1e-3 < mhh
	mask_JJ = data.M_JJ * 1e-3 < mhh

	#pHH_HH = pHH_HH[mask_HH];
	Y_HH = data.Y_HH[mask_HH];
	Z_HH = data.Z_HH[mask_HH];
	M_HH = data.M_HH[mask_HH];
	W_HH = data.W_HH[mask_HH];
	B_HH = data.HH[:,-10][mask_HH];

	#pttH_HH = pttH_HH[mask_ttH];
	Y_ttH = data.Y_ttH[mask_ttH];
	Z_ttH = data.Z_ttH[mask_ttH];
	M_ttH = data.M_ttH[mask_ttH];
	W_ttH = data.W_ttH[mask_ttH];
	B_ttH = data.ttH[:,-10][mask_ttH]; 
 
	#pZH_HH = pZH_HH[mask_ZH];
	Y_ZH = data.Y_ZH[mask_ZH];
	Z_ZH = data.Z_ZH[mask_ZH];
	M_ZH = data.M_ZH[mask_ZH];
	W_ZH = data.W_ZH[mask_ZH];
	B_ZH = data.ZH[:,-10][mask_ZH];

	#pJJ_HH = pJJ_HH[mask_JJ];
	Y_JJ = data.Y_JJ[mask_JJ];
	Z_JJ = data.Z_JJ[mask_JJ];
	M_JJ = data.M_JJ[mask_JJ];
	W_JJ = data.W_JJ[mask_JJ];
	B_JJ = data.JJ[:,-10][mask_JJ];

	for i in range(0,pHH_HH.shape[0]):
		if pHH_HH[i] > x1 and pHH_HH[i] < x2 and (Z_HH[i]*1e-3 < 130 and Z_HH[i]*1e-3 > 120) and (B_HH[i]*1e-3 < mbb_max and B_HH[i]*1e-3 > mbb_min):
			if cat == 5 and M_HH[i]*1e-3 < mhh:
				HH_yields.append(W_HH[i]*alpha)
				mbb.append(B_HH[i]*1e-3)
				myy.append(Z_HH[i]*1e-3)

	for i in range(0,pttH_HH.shape[0]):
		if pttH_HH[i] > x1 and pttH_HH[i] < x2 and (Z_ttH[i]*1e-3 < 130 and Z_ttH[i]*1e-3 > 120) and (B_ttH[i]*1e-3 < mbb_max and B_ttH[i]*1e-3 > mbb_min):
			if cat == 5 and M_ttH[i]*1e-3 < mhh:
				ttH_yields.append(W_ttH[i]*alpha)
			

	for i in range(0,pZH_HH.shape[0]):
		if pZH_HH[i] > x1 and pZH_HH[i] < x2 and (Z_ZH[i]*1e-3 < 130 and Z_ZH[i]*1e-3 > 120) and (B_ZH[i]*1e-3 < mbb_max and B_ZH[i]*1e-3 > mbb_min):
			if cat == 5 and M_ZH[i]*1e-3 < mhh:
				ZH_yields.append(W_ZH[i]*alpha)
			
			
	for i in range(0,pJJ_HH.shape[0]):
		if pJJ_HH[i] > x1 and pJJ_HH[i] < x2 and (Z_JJ[i]*1e-3 < 130 and Z_JJ[i]*1e-3 > 120)and (B_JJ[i]*1e-3 < mbb_max and B_JJ[i]*1e-3 > mbb_min):
			if cat == 5 and M_JJ[i]*1e-3 < mhh:
				JJ_yields.append(W_JJ[i]*alpha)

	return np.asarray(HH_yields), np.asarray(ttH_yields), np.asarray(ZH_yields), np.asarray(JJ_yields), np.asarray(mbb), np.asarray(myy)

def Compute_CutBased(cat):

	HH_yields  = []
	ttH_yields = []
	ZH_yields  = []
	JJ_yields  = []

	for i in range(0, data.C_HH.shape[0]):
		if data.C_HH[i]%10 == cat and (data.Z_HH[i]*1e-3 < 130 and data.Z_HH[i]*1e-3 > 120) and (data.HH[i,-10]*1e-3 < 140 and data.HH[i,-10]*1e-3 > 90):
			HH_yields.append(data.W_HH[i]*alpha)

	for i in range(0, data.C_ZH.shape[0]):
		if data.C_ZH[i]%10 == cat and (data.Z_ZH[i]*1e-3 < 130 and data.Z_ZH[i]*1e-3 > 120) and (data.ZH[i,-10]*1e-3 < 140 and data.ZH[i,-10]*1e-3 > 90):
			ZH_yields.append(data.W_ZH[i]*alpha)

	for i in range(0, data.C_ttH.shape[0]):
		if data.C_ttH[i]%10 == cat and (data.Z_ttH[i]*1e-3 < 130 and data.Z_ttH[i]*1e-3 > 120) and (data.ttH[i,-10]*1e-3 < 140 and data.ttH[i,-10]*1e-3 > 90):
			ttH_yields.append(data.W_ttH[i]*alpha)
			
	for i in range(0, data.C_JJ.shape[0]):
		if data.C_JJ[i]%10 == cat and (data.Z_JJ[i]*1e-3 < 130 and data.Z_JJ[i]*1e-3 > 120) and (data.JJ[i,-10]*1e-3 < 140 and data.JJ[i,-10]*1e-3 > 90):
			JJ_yields.append(data.W_JJ[i]*alpha)

	return np.asarray(HH_yields), np.asarray(ttH_yields), np.asarray(ZH_yields), np.asarray(JJ_yields)

def Scan_Z_High(HH,ttH,ZH,JJ,cat,y):

	Scale_HH  = [3.117, 4.090, 4.294, 4.303, alpha];  # SM
	#Scale_HH  = [3.984, 4.006, 3.912, 4.005, 3.993];  # +6
	#Scale_HH  = [4.014, 3.970, 4.081, 4.086, 4.008];  # +4
	#Scale_HH  = [4.119, 3.870, 4.018, 3.972, 3.984];  # +2
	#Scale_HH  = [3.876, 4.040, 3.991, 4.100, 4.038];  # -1
	#Scale_HH  = [4.124, 4.056, 3.989, 3.981, 4.007];  #  0
	#Scale_HH  = [4.065, 3.982, 3.891, 4.117, 4.031];  # -2
	#Scale_HH  = [4.010, 4.069, 3.937, 3.963, 3.998];  # -4
	#Scale_HH  = [3.849, 3.968, 4.110, 4.004, 3.980];  # -6
	Scale_ZH  = [4.177, 4.170, 4.270, 3.957, alpha];
	Scale_ttH = [4.042, 4.028, 3.970, 4.010, alpha];
	Scale_JJ  = [3.874, 4.245, 5.172, 3.397, alpha]; 

#	pHH_HH  = HH[:,0]
#	pttH_HH = ttH[:,0]
#	pZH_HH  = ZH[:,0]
#	pJJ_HH  = JJ[:,0]

	pHH_HH  = np.array(HH)
	pttH_HH = np.array(ttH)
	pZH_HH  = np.array(ZH)
	pJJ_HH  = np.array(JJ)

	mask_HH = data.M_HH * 1e-3 > mhh
	mask_ZH = data.M_ZH * 1e-3 > mhh
	mask_ttH = data.M_ttH * 1e-3 > mhh
	mask_JJ = data.M_JJ * 1e-3 > mhh

	pHH_HH = pHH_HH[mask_HH];
	Y_HH = data.Y_HH[mask_HH];
	Z_HH = data.Z_HH[mask_HH];
	M_HH = data.M_HH[mask_HH];
	W_HH = data.W_HH[mask_HH];
	B_HH = data.B_HH[mask_HH];

	pttH_HH = pttH_HH[mask_ttH];
	Y_ttH = data.Y_ttH[mask_ttH];
	Z_ttH = data.Z_ttH[mask_ttH];
	M_ttH = data.M_ttH[mask_ttH];
	W_ttH = data.W_ttH[mask_ttH];
	B_ttH = data.B_ttH[mask_ttH]; 
 
	pZH_HH = pZH_HH[mask_ZH];
	Y_ZH = data.Y_ZH[mask_ZH];
	Z_ZH = data.Z_ZH[mask_ZH];
	M_ZH = data.M_ZH[mask_ZH];
	W_ZH = data.W_ZH[mask_ZH];
	B_ZH = data.B_ZH[mask_ZH];

	pJJ_HH = pJJ_HH[mask_JJ];
	Y_JJ = data.Y_JJ[mask_JJ];
	Z_JJ = data.Z_JJ[mask_JJ];
	M_JJ = data.M_JJ[mask_JJ];
	W_JJ = data.W_JJ[mask_JJ];
	B_JJ = data.B_JJ[mask_JJ];


	X = np.linspace(-15., 5., 500)
	Sig = []
	dSig = []
	XX = []
	YY = []

	Z_max = 0;
	dZ_max = 0;
	cut = 0;
	for x in X:
		
		sumW_sig_unc = 0
		sumW_bkg_unc = 0
		sumW_sig = 0
		sumW_bkg = 0

		sumW_bkg_cut = 0
		for i in range(0,pHH_HH.shape[0]):
			if pHH_HH[i] > x and pHH_HH[i] < y and (Z_HH[i]*1e-3 < 130 and Z_HH[i]*1e-3 > 120) and (B_HH[i]*1e-3 < 140 and B_HH[i]*1e-3 > 90):
				if cat == 5 and M_HH[i]*1e-3 > mhh:
					sumW_sig = sumW_sig+W_HH[i]*alpha
					sumW_sig_unc = sumW_sig_unc+(W_HH[i]*alpha*W_HH[i]*alpha)
				
	
		for i in range(0,pttH_HH.shape[0]):
			if pttH_HH[i] > x and pttH_HH[i] < y and (Z_ttH[i]*1e-3 < 130 and Z_ttH[i]*1e-3 > 120) and (B_ttH[i]*1e-3 < 140 and B_ttH[i]*1e-3 > 90):
				if cat == 5 and M_ttH[i]*1e-3 > mhh:
					sumW_bkg = sumW_bkg+W_ttH[i]*alpha
					sumW_bkg_unc = sumW_bkg_unc+(W_ttH[i]*alpha*W_ttH[i]*alpha)
				

		for i in range(0,pZH_HH.shape[0]):
			if pZH_HH[i] > x and pZH_HH[i] < y and (Z_ZH[i]*1e-3 < 130 and Z_ZH[i]*1e-3 > 120) and (B_ZH[i]*1e-3 < 140 and B_ZH[i]*1e-3 > 90):
				if cat == 5 and M_ZH[i]*1e-3 > mhh:
					sumW_bkg = sumW_bkg+W_ZH[i]*alpha
					sumW_bkg_unc = sumW_bkg_unc+(W_ZH[i]*alpha*W_ZH[i]*alpha)
						
			
		for i in range(0,pJJ_HH.shape[0]):
			if pJJ_HH[i] > x and pJJ_HH[i] < y and (Z_JJ[i]*1e-3 < 130 and Z_JJ[i]*1e-3 > 120) and (B_JJ[i]*1e-3 < 140 and B_JJ[i]*1e-3 > 90):
				if cat == 5 and M_JJ[i]*1e-3 > mhh:
					sumW_bkg = sumW_bkg+W_JJ[i]*alpha
					sumW_bkg_unc = sumW_bkg_unc+(W_JJ[i]*alpha*W_JJ[i]*alpha)
					if (Z_JJ[i]*1e-3 < 127 and Z_JJ[i]*1e-3 > 123):
						sumW_bkg_cut = sumW_bkg_cut + W_JJ[i]*alpha
				
		if sumW_bkg < 0.0001 or sumW_bkg_cut < .8:
			continue;
		sumW_sig_unc = math.sqrt(sumW_sig_unc)
		sumW_bkg_unc = math.sqrt(sumW_bkg_unc)
		Z = math.sqrt( 2*( (sumW_sig+sumW_bkg)*math.log(1+(sumW_sig/sumW_bkg)) - sumW_sig ) )

		if Z < 0.001:
			continue;

		l=math.log(1+(sumW_sig/sumW_bkg));

		dZds = l/Z;
		dZdb = (l-sumW_sig/sumW_bkg)/Z;

		dZ=math.sqrt(dZds*dZds * sumW_sig_unc*sumW_sig_unc + dZdb*dZdb * sumW_bkg_unc*sumW_bkg_unc);
 
		Sig.append(Z)
		dSig.append(dZ)
		XX.append(x)

			

	return np.asarray(XX), np.asarray(Sig), np.asarray(dSig), np.ones(len(XX))*Sig[0], sumW_sig , sumW_bkg

def Scan_Z_Low(HH,ttH,ZH,JJ,cat,y):

	Scale_HH  = [3.117, 4.090, 4.294, 4.303, alpha];  # SM
	#Scale_HH  = [3.984, 4.006, 3.912, 4.005, 3.993];  # +6
	#Scale_HH  = [4.014, 3.970, 4.081, 4.086, 4.008];  # +4
	#Scale_HH  = [4.119, 3.870, 4.018, 3.972, 3.984];  # +2
	#Scale_HH  = [3.876, 4.040, 3.991, 4.100, 4.038];  # -1
	#Scale_HH  = [4.124, 4.056, 3.989, 3.981, 4.007];  #  0
	#Scale_HH  = [4.065, 3.982, 3.891, 4.117, 4.031];  # -2
	#Scale_HH  = [4.010, 4.069, 3.937, 3.963, 3.998];  # -4
	#Scale_HH  = [3.849, 3.968, 4.110, 4.004, 3.980];  # -6
	Scale_ZH  = [4.177, 4.170, 4.270, 3.957, alpha];
	Scale_ttH = [4.042, 4.028, 3.970, 4.010, alpha];
	Scale_JJ  = [3.874, 4.245, 5.172, 3.397, alpha]; 

#	pHH_HH  = HH[:,0]
#	pttH_HH = ttH[:,0]
#	pZH_HH  = ZH[:,0]
#	pJJ_HH  = JJ[:,0]

	pHH_HH  = np.array(HH)
	pttH_HH = np.array(ttH)
	pZH_HH  = np.array(ZH)
	pJJ_HH  = np.array(JJ)

	mask_HH = data.M_HH * 1e-3 < mhh
	mask_ZH = data.M_ZH * 1e-3 < mhh
	mask_ttH = data.M_ttH * 1e-3 < mhh
	mask_JJ = data.M_JJ * 1e-3 < mhh

	pHH_HH = pHH_HH[mask_HH];
	Y_HH = data.Y_HH[mask_HH];
	Z_HH = data.Z_HH[mask_HH];
	M_HH = data.M_HH[mask_HH];
	W_HH = data.W_HH[mask_HH];
	B_HH = data.B_HH[mask_HH];

	pttH_HH = pttH_HH[mask_ttH];
	Y_ttH = data.Y_ttH[mask_ttH];
	Z_ttH = data.Z_ttH[mask_ttH];
	M_ttH = data.M_ttH[mask_ttH];
	W_ttH = data.W_ttH[mask_ttH];
	B_ttH = data.B_ttH[mask_ttH]; 
 
	pZH_HH = pZH_HH[mask_ZH];
	Y_ZH = data.Y_ZH[mask_ZH];
	Z_ZH = data.Z_ZH[mask_ZH];
	M_ZH = data.M_ZH[mask_ZH];
	W_ZH = data.W_ZH[mask_ZH];
	B_ZH = data.B_ZH[mask_ZH];

	pJJ_HH = pJJ_HH[mask_JJ];
	Y_JJ = data.Y_JJ[mask_JJ];
	Z_JJ = data.Z_JJ[mask_JJ];
	M_JJ = data.M_JJ[mask_JJ];
	W_JJ = data.W_JJ[mask_JJ];
	B_JJ = data.B_JJ[mask_JJ];


	X = np.linspace(-15., 5., 500)
	Sig = []
	dSig = []
	XX = []
	YY = []

	Z_max = 0;
	dZ_max = 0;
	cut = 0;
	for x in X:
		
		sumW_sig_unc = 0
		sumW_bkg_unc = 0
		sumW_sig = 0
		sumW_bkg = 0

		sumW_bkg_cut = 0
		for i in range(0,pHH_HH.shape[0]):
			if pHH_HH[i] > x and pHH_HH[i] < y and (Z_HH[i]*1e-3 < 130 and Z_HH[i]*1e-3 > 120) and (B_HH[i]*1e-3 < 140 and B_HH[i]*1e-3 > 90):
				if cat == 5 and M_HH[i]*1e-3 < mhh:
					sumW_sig = sumW_sig+W_HH[i]*alpha
					sumW_sig_unc = sumW_sig_unc+(W_HH[i]*alpha*W_HH[i]*alpha)
				
	
		for i in range(0,pttH_HH.shape[0]):
			if pttH_HH[i] > x and pttH_HH[i] < y and (Z_ttH[i]*1e-3 < 130 and Z_ttH[i]*1e-3 > 120) and (B_ttH[i]*1e-3 < 140 and B_ttH[i]*1e-3 > 90):
				if cat == 5 and M_ttH[i]*1e-3 < mhh:
					sumW_bkg = sumW_bkg+W_ttH[i]*alpha
					sumW_bkg_unc = sumW_bkg_unc+(W_ttH[i]*alpha*W_ttH[i]*alpha)
				

		for i in range(0,pZH_HH.shape[0]):
			if pZH_HH[i] > x and pZH_HH[i] < y and (Z_ZH[i]*1e-3 < 130 and Z_ZH[i]*1e-3 > 120) and (B_ZH[i]*1e-3 < 140 and B_ZH[i]*1e-3 > 90):
				if cat == 5 and M_ZH[i]*1e-3 < mhh:
					sumW_bkg = sumW_bkg+W_ZH[i]*alpha
					sumW_bkg_unc = sumW_bkg_unc+(W_ZH[i]*alpha*W_ZH[i]*alpha)
						
			
		for i in range(0,pJJ_HH.shape[0]):
			if pJJ_HH[i] > x and pJJ_HH[i] < y and (Z_JJ[i]*1e-3 < 130 and Z_JJ[i]*1e-3 > 120) and (B_JJ[i]*1e-3 < 140 and B_JJ[i]*1e-3 > 90):
				if cat == 5 and M_JJ[i]*1e-3 < mhh:
					sumW_bkg = sumW_bkg+W_JJ[i]*alpha
					sumW_bkg_unc = sumW_bkg_unc+(W_JJ[i]*alpha*W_JJ[i]*alpha)
					if (Z_JJ[i]*1e-3 < 127 and Z_JJ[i]*1e-3 > 123):
						sumW_bkg_cut = sumW_bkg_cut + W_JJ[i]*alpha
				
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

			

	return np.asarray(XX), np.asarray(Sig), np.asarray(dSig), np.ones(len(XX))*Sig[0], sumW_sig , sumW_bkg

def Scan_Z2(HH,ttH,ZH,JJ,cat):

	Scale_HH  = [3.117, 4.090, 4.294, 4.303, 4.217];  # SM
	#Scale_HH  = [3.984, 4.006, 3.912, 4.005, 3.993];  # +6
	#Scale_HH  = [4.014, 3.970, 4.081, 4.086, 4.008];  # +4
	#Scale_HH  = [4.119, 3.870, 4.018, 3.972, 3.984];  # +2
	#Scale_HH  = [3.876, 4.040, 3.991, 4.100, 4.038];  # -1
	#Scale_HH  = [4.124, 4.056, 3.989, 3.981, 4.007];  #  0
	#Scale_HH  = [4.065, 3.982, 3.891, 4.117, 4.031];  # -2
	#Scale_HH  = [4.010, 4.069, 3.937, 3.963, 3.998];  # -4
	#Scale_HH  = [3.849, 3.968, 4.110, 4.004, 3.980];  # -6
	Scale_ZH  = [4.177, 4.170, 4.270, 3.957, 4.130];
	Scale_ttH = [4.042, 4.028, 3.970, 4.010, 4.022];
	Scale_JJ  = [3.874, 4.245, 5.172, 3.397, 3.970]; 

	pHH_HH  = HH[:,0]
	pttH_HH = ttH[:,0]
	pZH_HH  = ZH[:,0]
	pJJ_HH  = JJ[:,0]

	X = np.linspace(0., .99, 100)
	Sig = []
	dSig = []
	XX = []
	YY = []
	
	y=100;

	Z_max = 0;
	dZ_max = 0;
	cut = 0;
	for x in X:
		
		sumW_sig_unc = 0
		sumW_bkg_unc = 0
		sumW_sig = 0
		sumW_bkg = 0
		for i in range(0,pHH_HH.shape[0]):
			if pHH_HH[i] > x and pHH_HH[i] < y and (data.Z_HH[i]*1e-3 < 130 and data.Z_HH[i]*1e-3 > 120):
				if cat == 5 :
					sumW_sig = sumW_sig+data.W_HH[i]*alpha
					sumW_sig_unc = sumW_sig_unc+(data.W_HH[i]*alpha*data.W_HH[i]*alpha)
				elif data.C_HH[i] == cat:
					sumW_sig = sumW_sig+data.W_HH[i]*alpha
					sumW_sig_unc = sumW_sig_unc+(data.W_HH[i]*alpha*data.W_HH[i]*alpha)
	
		for i in range(0,pttH_HH.shape[0]):
			if pttH_HH[i] > x and pttH_HH[i] < y and (data.Z_ttH[i]*1e-3 < 130 and data.Z_ttH[i]*1e-3 > 120):
				if cat == 5 :
					sumW_bkg = sumW_bkg+data.W_ttH[i]*alpha
					sumW_bkg_unc = sumW_bkg_unc+(data.W_ttH[i]*alpha*data.W_ttH[i]*alpha)
				elif data.C_ttH[i] == cat:
					sumW_bkg = sumW_bkg+data.W_ttH[i]*alpha
					sumW_bkg_unc = sumW_bkg_unc+(data.W_ttH[i]*alpha*data.W_ttH[i]*alpha)

		for i in range(0,pZH_HH.shape[0]):
			if pZH_HH[i] > x and pZH_HH[i] < y and (data.Z_ZH[i]*1e-3 < 130 and data.Z_ZH[i]*1e-3 > 120):
				if cat == 5 :
					sumW_bkg = sumW_bkg+data.W_ZH[i]*alpha
					sumW_bkg_unc = sumW_bkg_unc+(data.W_ZH[i]*alpha*data.W_ZH[i]*alpha)
				elif data.C_ZH[i] == cat:
					sumW_bkg = sumW_bkg+data.W_ZH[i]*alpha
					sumW_bkg_unc = sumW_bkg_unc+(data.W_ZH[i]*alpha*data.W_ZH[i]*alpha)		
			
		for i in range(0,pJJ_HH.shape[0]):
			if pJJ_HH[i] > x and pJJ_HH[i] < y and (data.Z_JJ[i]*1e-3 < 130 and data.Z_JJ[i]*1e-3 > 120):
				if cat == 5 :
					sumW_bkg = sumW_bkg+data.W_JJ[i]*alpha
					sumW_bkg_unc = sumW_bkg_unc+(data.W_JJ[i]*alpha*data.W_JJ[i]*alpha)
				elif data.C_JJ[i] == cat:
					sumW_bkg = sumW_bkg+data.W_JJ[i]*alpha
					sumW_bkg_unc = sumW_bkg_unc+(data.W_JJ[i]*alpha*data.W_JJ[i]*alpha)
				
		if sumW_bkg < 0.001:
			continue;
		sumW_sig_unc = math.sqrt(sumW_sig_unc)
		sumW_bkg_unc = math.sqrt(sumW_bkg_unc)
		Z = math.sqrt( 2*( (sumW_sig+sumW_bkg)*math.log(1+(sumW_sig/sumW_bkg)) - sumW_sig ) )

		if Z < 0.001:
			continue;

		l=math.log(1+(sumW_sig/sumW_bkg));

		dZds = l/Z;
		dZdb = (l-sumW_sig/sumW_bkg)/Z;

		dZ=math.sqrt(dZds*dZds * sumW_sig_unc*sumW_sig_unc + dZdb*dZdb * sumW_bkg_unc*sumW_bkg_unc);
 
		Sig.append(Z)
		dSig.append(dZ)
		XX.append(x)

			

	return np.asarray(XX), np.asarray(Sig), np.asarray(dSig), np.ones(len(XX))*Sig[0]

def Scan_DHH(HH,ttH,ZH,JJ,cat):

	#Scale_HH  = [3.117, 4.090, 4.294, 4.303, 4.217];
	Scale_HH  = [3.984, 4.006, 3.912, 4.005, 3.993];
	Scale_ZH  = [4.177, 4.170, 4.270, 3.957, 4.130];
	Scale_ttH = [4.042, 4.028, 3.970, 4.010, 4.022];
	Scale_JJ  = [3.874, 4.245, 5.172, 3.397, 3.970];

	pHH_HH  = HH[:,0]
	pHH_ttH = HH[:,1]
	pHH_ZH  = HH[:,2]
	pHH_JJ  = HH[:,3]

	pttH_HH  = ttH[:,0]
	pttH_ttH = ttH[:,1]
	pttH_ZH  = ttH[:,2]
	pttH_JJ  = ttH[:,3]

	pZH_HH  = ZH[:,0]
	pZH_ttH = ZH[:,1]
	pZH_ZH  = ZH[:,2]
	pZH_JJ  = ZH[:,3]

	pJJ_HH  = JJ[:,0]
	pJJ_ttH = JJ[:,1]
	pJJ_ZH  = JJ[:,2]
	pJJ_JJ  = JJ[:,3]

	d_HH   = np.log( pHH_HH  / (pHH_ttH+pHH_ZH+pHH_JJ) )
	d_ttH  = np.log( pttH_HH / (pttH_ttH+pttH_ZH+pttH_JJ) )
	d_ZH   = np.log( pZH_HH  / (pZH_ttH+pZH_ZH+pZH_JJ) )
	d_JJ   = np.log( pJJ_HH  / (pJJ_ttH+pJJ_ZH+pJJ_JJ) )

	X = np.linspace(-30, 5, 350)
	Sig = []
	dSig = []
	DHH = []

	Z_max = 0;
	cut = 0
	for x in X:
		sumW_sig_unc = 0
		sumW_bkg_unc = 0
		sumW_sig = 0
		sumW_bkg = 0
		for i in range(0,d_HH.shape[0]):
			if d_HH[i] > x and (data.Z_HH[i]*1e-3 < 130 and data.Z_HH[i]*1e-3 > 120):
				if cat == 5:
					sumW_sig = sumW_sig+data.W_HH[i]*alpha;
					sumW_sig_unc = sumW_sig_unc+(data.W_HH[i]*alpha*data.W_HH[i]*alpha)
				elif data.C_HH[i] == cat:
					sumW_sig = sumW_sig+data.W_HH[i]*alpha;
					sumW_sig_unc = sumW_sig_unc+(data.W_HH[i]*alpha*data.W_HH[i]*alpha)

		for i in range(0,d_ttH.shape[0]):
			if d_ttH[i] > x and (data.Z_ttH[i]*1e-3 < 130 and data.Z_ttH[i]*1e-3 > 120):
				if cat == 5:
					sumW_bkg = sumW_bkg+data.W_ttH[i]*alpha;
					sumW_bkg_unc = sumW_bkg_unc+(data.W_ttH[i]*alpha*data.W_ttH[i]*alpha)
				elif data.C_ttH[i] == cat:
					sumW_bkg = sumW_bkg+data.W_ttH[i]*alpha;
					sumW_bkg_unc = sumW_bkg_unc+(data.W_ttH[i]*alpha*data.W_ttH[i]*alpha)

		for i in range(0,d_ZH.shape[0]):
			if d_ZH[i] > x and (data.Z_ZH[i]*1e-3 < 130 and data.Z_ZH[i]*1e-3 > 120):
				if cat == 5:
					sumW_bkg = sumW_bkg+data.W_ZH[i]*alpha;
					sumW_bkg_unc = sumW_bkg_unc+(data.W_ZH[i]*alpha*data.W_ZH[i]*alpha)
				elif data.C_ZH[i] == cat:
					sumW_bkg = sumW_bkg+data.W_ZH[i]*alpha;
					sumW_bkg_unc = sumW_bkg_unc+(data.W_ZH[i]*alpha*data.W_ZH[i]*alpha)

		for i in range(0,d_JJ.shape[0]):
			if d_JJ[i] > x and (data.Z_JJ[i]*1e-3 < 130 and data.Z_JJ[i]*1e-3 > 120):
				if cat == 5:
					sumW_bkg = sumW_bkg+data.W_JJ[i]*alpha;
					sumW_bkg_unc = sumW_bkg_unc+(data.W_JJ[i]*alpha*data.W_JJ[i]*alpha)
				elif data.C_JJ[i] == cat:
					sumW_bkg = sumW_bkg+data.W_JJ[i]*alpha;
					sumW_bkg_unc = sumW_bkg_unc+(data.W_JJ[i]*alpha*data.W_JJ[i]*alpha)
		if sumW_bkg < 0.001:
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
		DHH.append(x)

	return np.asarray(DHH), np.asarray(Sig), np.asarray(dSig), np.ones(len(DHH))*Sig[0]




def Apply_DHH_Cut(HH,ttH,ZH,JJ,cat,cut):

	#Scale_HH  = [3.117, 4.090, 4.294, 4.303, 4.217];
	Scale_HH  = [3.984, 4.006, 3.912, 4.005, 3.993];
	Scale_ZH  = [4.177, 4.170, 4.270, 3.957, 4.130];
	Scale_ttH = [4.042, 4.028, 3.970, 4.010, 4.022];
	Scale_JJ  = [3.874, 4.245, 5.172, 3.397, 3.970];

	pHH_HH  = HH[:,0]
	pHH_ttH = HH[:,1]
	pHH_ZH  = HH[:,2]
	pHH_JJ  = HH[:,3]

	pttH_HH  = ttH[:,0]
	pttH_ttH = ttH[:,1]
	pttH_ZH  = ttH[:,2]
	pttH_JJ  = ttH[:,3]

	pZH_HH  = ZH[:,0]
	pZH_ttH = ZH[:,1]
	pZH_ZH  = ZH[:,2]
	pZH_JJ  = ZH[:,3]

	pJJ_HH  = JJ[:,0]
	pJJ_ttH = JJ[:,1]
	pJJ_ZH  = JJ[:,2]
	pJJ_JJ  = JJ[:,3]

	d_HH   = np.log( pHH_HH  / (pHH_ttH+pHH_ZH+pHH_JJ) )
	d_ttH  = np.log( pttH_HH / (pttH_ttH+pttH_ZH+pttH_JJ) )
	d_ZH   = np.log( pZH_HH  / (pZH_ttH+pZH_ZH+pZH_JJ) )
	d_JJ   = np.log( pJJ_HH  / (pJJ_ttH+pJJ_ZH+pJJ_JJ) )

	x = cut
	HH_yields  = []
	ttH_yields = []
	ZH_yields  = []
	JJ_yields  = []

	for i in range(0,d_HH.shape[0]):
		if d_HH[i] > x and (data.Z_HH[i]*1e-3 < 130 and data.Z_HH[i]*1e-3 > 120):
			if cat == 5:
				HH_yields.append(data.W_HH[i]*alpha)
			elif data.C_HH[i] == cat:
				HH_yields.append(data.W_HH[i]*alpha)

	for i in range(0,d_ttH.shape[0]):
		if d_ttH[i] > x and (data.Z_ttH[i]*1e-3 < 130 and data.Z_ttH[i]*1e-3 > 120):
			if cat == 5:
				ttH_yields.append(data.W_ttH[i]*alpha)
			elif data.C_ttH[i] == cat:
				ttH_yields.append(data.W_ttH[i]*alpha)

	for i in range(0,d_ZH.shape[0]):
		if d_ZH[i] > x and (data.Z_ZH[i]*1e-3 < 130 and data.Z_ZH[i]*1e-3 > 120):
			if cat == 5:
				ZH_yields.append(data.W_ZH[i]*alpha)
			elif data.C_ZH[i] == cat:
				ZH_yields.append(data.W_ZH[i]*alpha)

	for i in range(0,d_JJ.shape[0]):
		if d_JJ[i] > x and (data.Z_JJ[i]*1e-3 < 130 and data.Z_JJ[i]*1e-3 > 120):
			if cat == 5:
				JJ_yields.append(data.W_JJ[i]*alpha)
			elif data.C_JJ[i] == cat:
				JJ_yields.append(data.W_JJ[i]*alpha)

	return np.asarray(HH_yields), np.asarray(ttH_yields), np.asarray(ZH_yields), np.asarray(JJ_yields)

def bootstrapping(HH,ttH,ZH,JJ,cat,cut,N):

	Z  = []
	Z0 = []

	#Scale_HH  = [3.117, 4.090, 4.294, 4.303, 4.217];
	Scale_HH  = [3.984, 4.006, 3.912, 4.005, 3.993];
	Scale_ZH  = [4.177, 4.170, 4.270, 3.957, 4.130];
	Scale_ttH = [4.042, 4.028, 3.970, 4.010, 4.022];
	Scale_JJ  = [3.874, 4.245, 5.172, 3.397, 3.970];

	pHH_HH  = HH[:,0]
	pttH_HH = ttH[:,0]
	pZH_HH  = ZH[:,0]
	pJJ_HH  = JJ[:,0]

	x = cut

	for i in range(0,N):
		HH_P  = np.random.poisson(1,len(HH))
		ttH_P = np.random.poisson(1,len(ttH))
		ZH_P  = np.random.poisson(1,len(ZH))
		YY_P  = np.random.poisson(1,len(JJ))

		HH_Y  = []
		ttH_Y = []
		ZH_Y  = []
		YY_Y  = []

		HH_Y_0  = []
		ttH_Y_0 = []
		ZH_Y_0  = []
		YY_Y_0  = []

		for i in range(0,pHH_HH.shape[0]):
			if pHH_HH[i] > x and (data.Z_HH[i]*1e-3 < 130 and data.Z_HH[i]*1e-3 > 120):
				if cat == 5:
					HH_Y.append(data.W_HH[i]*alpha*HH_P[i])
				elif data.C_HH[i] == cat:
					HH_Y.append(data.W_HH[i]*alpha*HH_P[i])

			if pHH_HH[i] > 0 and (data.Z_HH[i]*1e-3 < 130 and data.Z_HH[i]*1e-3 > 120):
				if cat == 5:
					HH_Y_0.append(data.W_HH[i]*alpha*HH_P[i])
				elif data.C_HH[i] == cat:
					HH_Y_0.append(data.W_HH[i]*alpha*HH_P[i])

		for i in range(0,pttH_HH.shape[0]):
			if pttH_HH[i] > x and (data.Z_ttH[i]*1e-3 < 130 and data.Z_ttH[i]*1e-3 > 120):
				if cat == 5:
					ttH_Y.append(data.W_ttH[i]*alpha*ttH_P[i])
				elif data.C_ttH[i] == cat:
					ttH_Y.append(data.W_ttH[i]*alpha*ttH_P[i])
			
			if pttH_HH[i] > 0 and (data.Z_ttH[i]*1e-3 < 130 and data.Z_ttH[i]*1e-3 > 120):
				if cat == 5:
					ttH_Y_0.append(data.W_ttH[i]*alpha*ttH_P[i])
				elif data.C_ttH[i] == cat:
					ttH_Y_0.append(data.W_ttH[i]*alpha*ttH_P[i])

		for i in range(0,pZH_HH.shape[0]):
			if pZH_HH[i] > x and (data.Z_ZH[i]*1e-3 < 130 and data.Z_ZH[i]*1e-3 > 120):
				if cat == 5:
					ZH_Y.append(data.W_ZH[i]*alpha*ZH_P[i])
				elif data.C_ZH[i] == cat:
					ZH_Y.append(data.W_ZH[i]*alpha*ZH_P[i])
			
			if pZH_HH[i] > 0 and (data.Z_ZH[i]*1e-3 < 130 and data.Z_ZH[i]*1e-3 > 120):
				if cat == 5:
					ZH_Y_0.append(data.W_ZH[i]*alpha*ZH_P[i])
				elif data.C_ZH[i] == cat:
					ZH_Y_0.append(data.W_ZH[i]*alpha*ZH_P[i])

		for i in range(0,pJJ_HH.shape[0]):
			if pJJ_HH[i] > x and (data.Z_JJ[i]*1e-3 < 130 and data.Z_JJ[i]*1e-3 > 120):
				if cat == 5:
					YY_Y.append(data.W_JJ[i]*alpha*YY_P[i])
				elif data.C_JJ[i] == cat:
					YY_Y.append(data.W_JJ[i]*alpha*YY_P[i])
		
			if pJJ_HH[i] > 0 and (data.Z_JJ[i]*1e-3 < 130 and data.Z_JJ[i]*1e-3 > 120):
				if cat == 5:
					YY_Y_0.append(data.W_JJ[i]*alpha*YY_P[i])
				elif data.C_JJ[i] == cat:
					YY_Y_0.append(data.W_JJ[i]*alpha*YY_P[i])

			Sum_HH_Y  = np.sum(np.asarray(HH_Y))
			Sum_ttH_Y = np.sum(np.asarray(ttH_Y))
			Sum_ZH_Y  = np.sum(np.asarray(ZH_Y))
			Sum_YY_Y  = np.sum(np.asarray(YY_Y))

			Sum_HH_Y_0  = np.sum(np.asarray(HH_Y_0))
			Sum_ttH_Y_0 = np.sum(np.asarray(ttH_Y_0))
			Sum_ZH_Y_0  = np.sum(np.asarray(ZH_Y_0))
			Sum_YY_Y_0  = np.sum(np.asarray(YY_Y_0))

			sumW_sig = Sum_HH_Y
			sumW_bkg = Sum_ttH_Y+Sum_ZH_Y+Sum_YY_Y

			sumW_sig0 = Sum_HH_Y_0
			sumW_bkg0 = Sum_ttH_Y_0+Sum_ZH_Y_0+Sum_YY_Y_0

			z =  math.sqrt( 2*( (sumW_sig+sumW_bkg)*math.log(1+(sumW_sig/sumW_bkg)) - sumW_sig ) )
			z0 =  math.sqrt( 2*( (sumW_sig0+sumW_bkg0)*math.log(1+(sumW_sig0/sumW_bkg0)) - sumW_sig0 ) )
		
			Z.append(z)
			Z0.append(z0)
	return np.asarray(Z), np.asarray(Z0)
		
def discriminant(HH,ttH,ZH,JJ):

	pHH_HH  = HH[:,0]
	pHH_ttH = HH[:,1]
	pHH_ZH  = HH[:,2]
	pHH_JJ  = HH[:,3]

	pttH_HH  = ttH[:,0]
	pttH_ttH = ttH[:,1]
	pttH_ZH  = ttH[:,2]
	pttH_JJ  = ttH[:,3]

	pZH_HH  = ZH[:,0]
	pZH_ttH = ZH[:,1]
	pZH_ZH  = ZH[:,2]
	pZH_JJ  = ZH[:,3]

	pJJ_HH  = JJ[:,0]
	pJJ_ttH = JJ[:,1]
	pJJ_ZH  = JJ[:,2]
	pJJ_JJ  = JJ[:,3]

	d_HH   = np.log( pHH_HH  / (pHH_ttH+pHH_ZH+pHH_JJ) )
	d_ttH  = np.log( pttH_HH / (pttH_ttH+pttH_ZH+pttH_JJ) )
	d_ZH   = np.log( pZH_HH  / (pZH_ttH+pZH_ZH+pZH_JJ) )
	d_JJ   = np.log( pJJ_HH  / (pJJ_ttH+pJJ_ZH+pJJ_JJ) )

	return d_HH, d_ttH, d_ZH, d_JJ

def Scan_BSM_SM(d_HH, d_ttH, d_ZH, d_JJ, cat, k):

	Scale_HH  = [3.117, 4.090, 4.294, 4.303, 4.217];
	Scale_HH_k6  = [3.984, 4.006, 3.912, 4.005, 3.993];
	Scale_ZH  = [4.177, 4.170, 4.270, 3.957, 4.130];
	Scale_ttH = [4.042, 4.028, 3.970, 4.010, 4.022];
	Scale_JJ  = [3.874, 4.245, 5.172, 3.397, 3.970];


	if k == 1:
		Scale_HH = Scale_HH
	elif k == 6:
		Scale_HH = Scale_HH_k6


	X = np.linspace(-30, 5, 350)
	Sig = []
	dSig = []
	DHH = []

	Z_max = 0;
	cut = 0
	for x in X:
		sumW_sig_unc = 0
		sumW_bkg_unc = 0
		sumW_sig = 0
		sumW_bkg = 0
		for i in range(0,d_HH.shape[0]):
			if d_HH[i] > x and (data.Z_HH[i]*1e-3 < 130 and data.Z_HH[i]*1e-3 > 120):
				if cat == 5:
					sumW_sig = sumW_sig+data.W_HH[i]*alpha;
					sumW_sig_unc = sumW_sig_unc+(data.W_HH[i]*alpha*data.W_HH[i]*alpha)
				elif data.C_HH[i] == cat:
					sumW_sig = sumW_sig+data.W_HH[i]*alpha;
					sumW_sig_unc = sumW_sig_unc+(data.W_HH[i]*alpha*data.W_HH[i]*alpha)

		for i in range(0,d_ttH.shape[0]):
			if d_ttH[i] > x and (data.Z_ttH[i]*1e-3 < 130 and data.Z_ttH[i]*1e-3 > 120):
				if cat == 5:
					sumW_bkg = sumW_bkg+data.W_ttH[i]*alpha;
					sumW_bkg_unc = sumW_bkg_unc+(data.W_ttH[i]*alpha*data.W_ttH[i]*alpha)
				elif data.C_ttH[i] == cat:
					sumW_bkg = sumW_bkg+data.W_ttH[i]*alpha;
					sumW_bkg_unc = sumW_bkg_unc+(data.W_ttH[i]*alpha*data.W_ttH[i]*alpha)

		for i in range(0,d_ZH.shape[0]):
			if d_ZH[i] > x and (data.Z_ZH[i]*1e-3 < 130 and data.Z_ZH[i]*1e-3 > 120):
				if cat == 5:
					sumW_bkg = sumW_bkg+data.W_ZH[i]*alpha;
					sumW_bkg_unc = sumW_bkg_unc+(data.W_ZH[i]*alpha*data.W_ZH[i]*alpha)
				elif data.C_ZH[i] == cat:
					sumW_bkg = sumW_bkg+data.W_ZH[i]*alpha;
					sumW_bkg_unc = sumW_bkg_unc+(data.W_ZH[i]*alpha*data.W_ZH[i]*alpha)

		for i in range(0,d_JJ.shape[0]):
			if d_JJ[i] > x and (data.Z_JJ[i]*1e-3 < 130 and data.Z_JJ[i]*1e-3 > 120):
				if cat == 5:
					sumW_bkg = sumW_bkg+data.W_JJ[i]*alpha;
					sumW_bkg_unc = sumW_bkg_unc+(data.W_JJ[i]*alpha*data.W_JJ[i]*alpha)
				elif data.C_JJ[i] == cat:
					sumW_bkg = sumW_bkg+data.W_JJ[i]*alpha;
					sumW_bkg_unc = sumW_bkg_unc+(data.W_JJ[i]*alpha*data.W_JJ[i]*alpha)
		if sumW_bkg < 0.001:
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
		DHH.append(x)

	return np.asarray(DHH), np.asarray(Sig), np.asarray(dSig), np.ones(len(DHH))*Sig[0]

def ComputeZ(sig,bkg,unc_sig,unc_bkg):

	Z0 = math.sqrt( 2*( (sig+bkg)*math.log(1+(sig/bkg)) - sig ) )
	l = math.log(1+ (sig/bkg))
	dZds = l/Z0
	dZdb = (l-(sig/bkg))/Z0
	dZ = math.sqrt((dZdb*unc_bkg)**2+(dZds*unc_sig)**2)
	return Z0 , dZ


def DHH(X,Y,Z,W,process):

	if process == 'SM':
		return [math.log(X[i]*8.8111e-5/(Y[i]*0.00114975+Z[i]*0.00172452+W[i]*51.823)) for i in range(0,X.shape[0])]

	if process == 'BSM':
		return [math.log(X[i]*0.001711/(Y[i]*0.00114975+Z[i]*0.00172452+W[i]*51.823)) for i in range(0,X.shape[0])]
	

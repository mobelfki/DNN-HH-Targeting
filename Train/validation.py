#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from DataProcessing import Data
from keras import optimizers
from keras.models import model_from_json
from sklearn.metrics import roc_curve, roc_auc_score, auc
from scipy import interp
from itertools import cycle
from tools import Scan_Z_High, Scan_DHH, Apply_Z_Cut_High, Apply_Z_Cut_Low, Apply_DHH_Cut, bootstrapping, discriminant, Scan_Z2, ComputeZ, Scan_Z_Low, Compute_CutBased, DHH

import os
import math


Kappa = 1;

data = Data(Kappa,'NLO');


SM_dir_name = 'SM_model_Mbb_NLO_2btag77/'
SM_model_name = 'SM_model_Mbb_NLO_2btag77' 

SM_json_name = SM_model_name+'.json'
SM_h5_weight = SM_model_name+'.h5'

SM_json_file = open(SM_dir_name+SM_json_name,'r');

SM_model_json = SM_json_file.read();
SM_json_file.close();
SM_model = model_from_json(SM_model_json);
SM_model.load_weights(SM_dir_name+SM_h5_weight);

adm = optimizers.Adam(lr=0.00001)
SM_model.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['categorical_accuracy']);


BSM_dir_name = 'BSM_model_Mbb_NLO_2btag77/'
BSM_model_name = 'BSM_model_Mbb_NLO_2btag77' 

BSM_json_name = BSM_model_name+'.json'
BSM_h5_weight = BSM_model_name+'.h5'

BSM_json_file = open(BSM_dir_name+BSM_json_name,'r');

BSM_model_json = BSM_json_file.read();
BSM_json_file.close();
BSM_model = model_from_json(BSM_model_json);
BSM_model.load_weights(BSM_dir_name+BSM_h5_weight);

adm = optimizers.Adam(lr=0.00001)
BSM_model.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['categorical_accuracy']);


"""
model = SM_model

HH   = model.predict(data.X_HH)
ttH  = model.predict(data.X_ttH)
ZH   = model.predict(data.X_ZH)
JJ   = model.predict(data.X_JJ)

pHH = DHH(HH[:,0], HH[:,1], HH[:,2], HH[:,3], 'SM')
pttH = DHH(ttH[:,0], ttH[:,1], ttH[:,2], ttH[:,3], 'SM')
pZH = DHH(ZH[:,0], ZH[:,1], ZH[:,2], ZH[:,3], 'SM')
pJJ = DHH(JJ[:,0], JJ[:,1], JJ[:,2], JJ[:,3], 'SM')


eff_5, Z_5, dZ_5, Z0_5, sig, bkg = Scan_Z_High(pHH,pttH,pZH,pJJ,5,100);

print('Z max = %.3f +/- %.3f ; x = %.3f ' %(np.max(Z_5), dZ_5[np.argmax(Z_5)], eff_5[np.argmax(Z_5)]))
print(' Sig = %.3f ; Bkg %.3f ' %(sig,bkg))

eff_5, Z_5, dZ_5, Z0_5, sig, bkg = Scan_Z_High(pHH,pttH,pZH,pJJ,5,eff_5[np.argmax(Z_5)]);

print('Z min = %.3f +/- %.3f ; x = %.3f ' %(np.max(Z_5), dZ_5[np.argmax(Z_5)], eff_5[np.argmax(Z_5)]))
print(' Sig = %.3f ; Bkg %.3f ' %(sig,bkg))

exit()


model = BSM_model


HH   = model.predict(data.X_HH)
ttH  = model.predict(data.X_ttH)
ZH   = model.predict(data.X_ZH)
JJ   = model.predict(data.X_JJ)

pHH = DHH(HH[:,0], HH[:,1], HH[:,2], HH[:,3], 'BSM')
pttH = DHH(ttH[:,0], ttH[:,1], ttH[:,2], ttH[:,3], 'BSM')
pZH = DHH(ZH[:,0], ZH[:,1], ZH[:,2], ZH[:,3], 'BSM')
pJJ = DHH(JJ[:,0], JJ[:,1], JJ[:,2], JJ[:,3], 'BSM')


eff_5, Z_5, dZ_5, Z0_5, sig, bkg = Scan_Z_Low(pHH,pttH,pZH,pJJ,5,100);

print('Z max = %.3f +/- %.3f ; x = %.3f ' %(np.max(Z_5), dZ_5[np.argmax(Z_5)], eff_5[np.argmax(Z_5)]))
print(' Sig = %.3f ; Bkg %.3f ' %(sig,bkg))

eff_5, Z_5, dZ_5, Z0_5, sig, bkg = Scan_Z_Low(pHH,pttH,pZH,pJJ,5,eff_5[np.argmax(Z_5)]);

print('Z min = %.3f +/- %.3f ; x = %.3f ' %(np.max(Z_5), dZ_5[np.argmax(Z_5)], eff_5[np.argmax(Z_5)]))
print(' Sig = %.3f ; Bkg %.3f ' %(sig,bkg))

exit()
"""


mask_HH = data.M_HH * 1e-3 > 350
mask_ZH = data.M_ZH * 1e-3 > 350
mask_ttH = data.M_ttH * 1e-3 > 350
mask_JJ = data.M_JJ * 1e-3 > 350


HH   = SM_model.predict(data.X_HH[mask_HH])
ttH  = SM_model.predict(data.X_ttH[mask_ttH])
ZH   = SM_model.predict(data.X_ZH[mask_ZH])
JJ   = SM_model.predict(data.X_JJ[mask_JJ])

#r_high = [-10.98,100] #SM 85 NLO
#r_low  = [-12.59,-10.73] #SM 85 NLO

r_high = [-11.63,100] #SM 77 NLO
r_low  = [-13.59,-11.63] #SM 77 NLO


pHH = DHH(HH[:,0], HH[:,1], HH[:,2], HH[:,3], 'SM')
pttH = DHH(ttH[:,0], ttH[:,1], ttH[:,2], ttH[:,3], 'SM')
pZH = DHH(ZH[:,0], ZH[:,1], ZH[:,2], ZH[:,3], 'SM')
pJJ = DHH(JJ[:,0], JJ[:,1], JJ[:,2], JJ[:,3], 'SM')


hh, tth, zh, jj, mbb_high, myy_high = Apply_Z_Cut_High(pHH,pttH,pZH,pJJ,5,r_high)

sig = np.sum(hh)

sig_unc = math.sqrt( np.sum(hh*hh) )

bkg = np.sum(tth)+np.sum(zh)+np.sum(jj)

bkg_unc = math.sqrt( np.sum(tth*tth)+np.sum(zh*zh)+np.sum(jj*jj) )

Z1, dZ1 = ComputeZ(sig,bkg,sig_unc,bkg_unc)

print('High Mass , High pHH *** Z %.3f +/- %.3f'%(Z1,dZ1))
print(' Sig = %.3f ; Bkg %.3f ' %(sig,bkg))
print(' ttH = %.3f ; ZH = %.3f ; JJ = %.3f'%(np.sum(tth),np.sum(zh),np.sum(jj)) )

hh, tth, zh, jj, mbb_low, myy_low = Apply_Z_Cut_High(pHH,pttH,pZH,pJJ,5,r_low)

sig = np.sum(hh)

sig_unc = math.sqrt( np.sum(hh*hh) )

bkg = np.sum(tth)+np.sum(zh)+np.sum(jj)

bkg_unc = math.sqrt( np.sum(tth*tth)+np.sum(zh*zh)+np.sum(jj*jj) )

Z2, dZ2 = ComputeZ(sig,bkg,sig_unc,bkg_unc)

print('High Mass , Low pHH *** Z %.3f +/- %.3f'%(Z2,dZ2))
print(' Sig = %.3f ; Bkg %.3f ' %(sig,bkg))
print(' ttH = %.3f ; ZH = %.3f ; JJ = %.3f'%(np.sum(tth),np.sum(zh),np.sum(jj)) )


mask_HH = data.M_HH * 1e-3 < 350
mask_ZH = data.M_ZH * 1e-3 < 350
mask_ttH = data.M_ttH * 1e-3 < 350
mask_JJ = data.M_JJ * 1e-3 < 350

HH   = BSM_model.predict(data.X_HH[mask_HH])
ttH  = BSM_model.predict(data.X_ttH[mask_ttH])
ZH   = BSM_model.predict(data.X_ZH[mask_ZH])
JJ   = BSM_model.predict(data.X_JJ[mask_JJ])

#r_high = [-9.73,100] #BSM 85 NLO
#r_low  = [-10.78,-9.73] #BSM 85 NLO

r_high = [-8.35,100] #BSM 77 LO 6
r_low  = [-9.79,-8.35] #BSM 77 LO 6

r_high = [-9.11,100] #BSM 77 NLO 10
r_low  = [-10.31,-9.11] #BSM 77 NLO 10

pHH = DHH(HH[:,0], HH[:,1], HH[:,2], HH[:,3], 'BSM')
pttH = DHH(ttH[:,0], ttH[:,1], ttH[:,2], ttH[:,3], 'BSM')
pZH = DHH(ZH[:,0], ZH[:,1], ZH[:,2], ZH[:,3], 'BSM')
pJJ = DHH(JJ[:,0], JJ[:,1], JJ[:,2], JJ[:,3], 'BSM')


hh, tth, zh, jj, mbb_high, myy_high = Apply_Z_Cut_Low(pHH,pttH,pZH,pJJ,5,r_high)

sig = np.sum(hh)

sig_unc = math.sqrt( np.sum(hh*hh) )

bkg = np.sum(tth)+np.sum(zh)+np.sum(jj)

bkg_unc = math.sqrt( np.sum(tth*tth)+np.sum(zh*zh)+np.sum(jj*jj) )

Z3, dZ3 = ComputeZ(sig,bkg,sig_unc,bkg_unc)

print('Low Mass , High pHH *** Z %.3f +/- %.3f'%(Z3,dZ3))
print(' Sig = %.3f ; Bkg %.3f ' %(sig,bkg))
print(' ttH = %.3f ; ZH = %.3f ; JJ = %.3f'%(np.sum(tth),np.sum(zh),np.sum(jj)) )

hh, tth, zh, jj, mbb_low, myy_low = Apply_Z_Cut_Low(pHH,pttH,pZH,pJJ,5,r_low)

sig =  np.sum(hh)

sig_unc = math.sqrt( np.sum(hh*hh) )

bkg = np.sum(tth)+np.sum(zh)+np.sum(jj)

bkg_unc = math.sqrt( np.sum(tth*tth)+np.sum(zh*zh)+np.sum(jj*jj))


Z4, dZ4 = ComputeZ(sig,bkg,sig_unc,bkg_unc)

print('Low Mass , Low pHH *** Z %.3f +/- %.3f'%(Z4,dZ4))
print(' Sig = %.3f ; Bkg %.3f ' %(sig,bkg))
print(' ttH = %.3f ; ZH = %.3f ; JJ = %.3f'%(np.sum(tth),np.sum(zh),np.sum(jj)) )

Z = math.sqrt(Z1*Z1 + Z2*Z2 + Z3*Z3 + Z4*Z4)
dZ = (Z1/Z)*dZ1 + (Z2/Z)*dZ2 + (Z3/Z)*dZ3 + (Z4/Z)*dZ4

print('Combined %.3f +/- %.3f' %(Z,dZ))

exit()

BSM_ZH  = BSM_model.predict(data.X_ZH)
SM_ZH   = SM_model.predict(data.X_ZH)

BSM_H7_ZH  = BSM_model.predict(data.X_H7_ZH)
SM_H7_ZH   = SM_model.predict(data.X_H7_ZH)


BSM_ttH  = BSM_model.predict(data.X_ttH)
SM_ttH   = SM_model.predict(data.X_ttH)

BSM_H7_ttH  = BSM_model.predict(data.X_H7_ttH)
SM_H7_ttH   = SM_model.predict(data.X_H7_ttH)

D_BSM_ZH = DHH(BSM_ZH[:,0], BSM_ZH[:,1], BSM_ZH[:,2], BSM_ZH[:,3], 'BSM')
D_SM_ZH  = DHH(SM_ZH[:,0], SM_ZH[:,1], SM_ZH[:,2], SM_ZH[:,3], 'SM')

D_H7_BSM_ZH = DHH(BSM_H7_ZH[:,0], BSM_H7_ZH[:,1], BSM_H7_ZH[:,2], BSM_H7_ZH[:,3], 'BSM')
D_H7_SM_ZH  = DHH(SM_H7_ZH[:,0], SM_H7_ZH[:,1], SM_H7_ZH[:,2], SM_H7_ZH[:,3], 'SM')

D_BSM_ttH = DHH(BSM_ttH[:,0], BSM_ttH[:,1], BSM_ttH[:,2], BSM_ttH[:,3], 'BSM')
D_SM_ttH  = DHH(SM_ttH[:,0], SM_ttH[:,1], SM_ttH[:,2], SM_ttH[:,3], 'SM')

D_H7_BSM_ttH = DHH(BSM_H7_ttH[:,0], BSM_H7_ttH[:,1], BSM_H7_ttH[:,2], BSM_H7_ttH[:,3], 'BSM')
D_H7_SM_ttH  = DHH(SM_H7_ttH[:,0], SM_H7_ttH[:,1], SM_H7_ttH[:,2], SM_H7_ttH[:,3], 'SM')

fig, (ax1, ax2) = plt.subplots(nrows=2)

fig, (ax1, ax2) = plt.subplots(nrows=2)
n_ZH, bins, patches = ax1.hist(D_BSM_ZH, bins=range(-18,-5), histtype='step',   density='True', linewidth=1.5)
n_H7_ZH, bins, patches =ax1.hist(D_H7_BSM_ZH, bins=range(-18,-5), histtype='step',   density='True', linewidth=1.5)
ax2.scatter(bins[:-1], n_ZH / n_H7_ZH)
ax1.legend(['Pythia8','Herwig7'])
ax2.set_ylabel('Pythia8/Herwig7')
ax2.set_xlabel('BSM DHH')
ax2.set_ylim(0.5,1.5)
fig.savefig('D_BSM_ZH.png')

plt.figure()
fig, (ax1, ax2) = plt.subplots(nrows=2)
n_ZH , _, _ =ax1.hist(D_SM_ZH, bins=range(-20,-5), histtype='step',   density='True', linewidth=1.5)
n_H7_ZH, bins, _ = ax1.hist(D_H7_SM_ZH, bins=range(-20,-5), histtype='step',   density='True', linewidth=1.5)
ax1.legend(['Pythia8','Herwig7'])
ax2.scatter(bins[:-1], n_ZH / n_H7_ZH)
ax2.set_ylabel('Pythia8/Herwig7')
ax2.set_xlabel('SM DHH')
ax2.set_ylim(0.5,1.5)
fig.savefig('D_SM_ZH.png')


fig, (ax1, ax2) = plt.subplots(nrows=2)
n_ttH, bins, patches = ax1.hist(D_BSM_ttH, bins=range(-18,-5), histtype='step',   density='True', linewidth=1.5)
n_H7_ttH, bins, patches =ax1.hist(D_H7_BSM_ttH, bins=range(-18,-5), histtype='step',   density='True', linewidth=1.5)
ax2.scatter(bins[:-1], n_ttH / n_H7_ttH)
ax1.legend(['Pythia8','Herwig7'])
ax2.set_ylabel('Pythia8/Herwig7')
ax2.set_xlabel('BSM DHH')
ax2.set_ylim(0.5,1.5)
fig.savefig('D_BSM_ttH.png')

plt.figure()
fig, (ax1, ax2) = plt.subplots(nrows=2)
n_ttH , _, _ =ax1.hist(D_SM_ttH, bins=range(-20,-5), histtype='step',   density='True', linewidth=1.5)
n_H7_ttH, bins, _ = ax1.hist(D_H7_SM_ttH, bins=range(-20,-5), histtype='step',   density='True', linewidth=1.5)
ax1.legend(['Pythia8','Herwig7'])
ax2.scatter(bins[:-1], n_ttH / n_H7_ttH)
ax2.set_ylabel('Pythia8/Herwig7')
ax2.set_xlabel('SM DHH')
ax2.set_ylim(0.5,1.5)
fig.savefig('D_SM_ttH.png')


mask_HH  = data.M_HH * 1e-3 < 350
LM_HH    = data.X_HH[mask_HH]
LM_W_HH  = data.W_HH[mask_HH]
mask_HH  = data.M_HH * 1e-3 >= 350
mask_HH1 = data.Z_HH * 1e-3 > 120
mask_HH2 = data.Z_HH * 1e-3 < 130
mask_HH3 = data.HH[:,-10] * 1e-3 > 90
mask_HH4 = data.HH[:,-10] * 1e-3 < 140
mask_HH = mask_HH * mask_HH1 * mask_HH2 * mask_HH3 * mask_HH4
MBB = data.HH[:,-10][mask_HH]
MYY = data.Z_HH[mask_HH]
HM_HH = data.X_HH[mask_HH]
HM_W_HH = data.W_HH[mask_HH]

mask_ttH = data.M_ttH * 1e-3 < 350
LM_ttH = data.X_ttH[mask_ttH]
LM_W_ttH = data.W_ttH[mask_ttH]
mask_ttH = data.M_ttH * 1e-3 >= 350
HM_ttH = data.X_ttH[mask_ttH]
HM_W_ttH = data.W_ttH[mask_ttH]

mask_ZH = data.M_ZH * 1e-3 < 350
LM_ZH = data.X_ZH[mask_ZH]
LM_W_ZH = data.W_ZH[mask_ZH]
mask_ZH = data.M_ZH * 1e-3 >= 350
HM_ZH = data.X_ZH[mask_ZH]
HM_W_ZH = data.W_ZH[mask_ZH]

mask_JJ = data.M_JJ * 1e-3 < 350
LM_JJ = data.X_JJ[mask_JJ]
LM_W_JJ = data.W_JJ[mask_JJ]
mask_JJ = data.M_JJ * 1e-3 >= 350
mask_JJ1 = data.Z_JJ * 1e-3 > 120
mask_JJ2 = data.Z_JJ * 1e-3 < 130
mask_JJ3 = data.JJ[:,-10] * 1e-3 > 90
mask_JJ4 = data.JJ[:,-10] * 1e-3 < 140
#mask_JJ = mask_JJ * mask_JJ1 * mask_JJ2 * mask_JJ3 * mask_JJ4
JJ_Y = data.Z_JJ[mask_JJ]
JJ_B = data.JJ[:,-10][mask_JJ]
HM_JJ = data.X_JJ[mask_JJ]
HM_W_JJ = data.W_JJ[mask_JJ]

data2 = Data(10,'NLO');

mask_HH_6 = data2.M_HH * 1e-3 < 350
LM_HH_6 = data2.X_HH[mask_HH_6]
LM_W_HH_6 = data2.W_HH[mask_HH_6]
mask_HH_6 = data2.M_HH * 1e-3 >= 350
HM_HH_6 = data2.X_HH[mask_HH_6]
HM_W_HH_6 = data2.W_HH[mask_HH_6]

mask_Data = data.M_Data * 1e-3 < 350
mask2_Data = data.Z_Data * 1e-3 < 120
mask3_Data = data.Z_Data * 1e-3 > 130
mask_Data = mask_Data * (mask2_Data + mask3_Data)
LM_Data = data.X_Data[mask_Data]
mask_Data = data.M_Data * 1e-3 >= 350
mask2_Data = data.Z_Data * 1e-3 < 120
mask3_Data = data.Z_Data * 1e-3 > 130
mask_Data = mask_Data * (mask2_Data + mask3_Data)
HM_Data = data.X_Data[mask_Data]

B_HH  = BSM_model.predict(data.X_HH)
B_ttH = BSM_model.predict(data.X_ttH)
B_ZH  = BSM_model.predict(data.X_ZH)
B_JJ  = BSM_model.predict(data.X_JJ)

S_HH  = SM_model.predict(data.X_HH)
S_ttH = SM_model.predict(data.X_ttH)
S_ZH  = SM_model.predict(data.X_ZH)
S_JJ  = SM_model.predict(data.X_JJ)


dHH_HM  = DHH(S_HH[:,0],  S_HH[:,1],  S_HH[:,2],  S_HH[:,3], 'SM')
dttH_HM = DHH(S_ttH[:,0], S_ttH[:,1], S_ttH[:,2], S_ttH[:,3], 'SM')
dZH_HM  = DHH(S_ZH[:,0],  S_ZH[:,1],  S_ZH[:,2],  S_ZH[:,3], 'SM')
dJJ_HM  = DHH(S_JJ[:,0],  S_JJ[:,1],  S_JJ[:,2],  S_JJ[:,3], 'SM')


dHH_LM  = DHH(B_HH[:,0],  B_HH[:,1],  B_HH[:,2],  B_HH[:,3], 'BSM')
dttH_LM = DHH(B_ttH[:,0], B_ttH[:,1], B_ttH[:,2], B_ttH[:,3], 'BSM')
dZH_LM  = DHH(B_ZH[:,0],  B_ZH[:,1],  B_ZH[:,2],  B_ZH[:,3], 'BSM')
dJJ_LM  = DHH(B_JJ[:,0],  B_JJ[:,1],  B_JJ[:,2],  B_JJ[:,3], 'BSM')

plt.figure()
plt.hist2d(dHH_HM, dHH_LM, bins=50, range=[[-30,5],[-30,5]])
plt.xlabel('dHH_SM')
plt.ylabel('dHH_BSM')
plt.savefig('dHH_SM_BSM.png')

plt.figure()
plt.hist2d(dZH_HM, dZH_LM, bins=50, range=[[-30,5],[-30,5]])
plt.xlabel('dZH_SM')
plt.ylabel('dZH_BSM')
plt.savefig('dZH_SM_BSM.png')

plt.figure()
plt.hist2d(dttH_HM, dttH_LM, bins=50, range=[[-30,5],[-30,5]])
plt.xlabel('dttH_SM')
plt.ylabel('dttH_BSM')
plt.savefig('dttH_SM_BSM.png')

plt.figure()
plt.hist2d(dJJ_HM, dJJ_LM, bins=50, range=[[-30,5],[-30,5]])
plt.xlabel('dJJ_SM')
plt.ylabel('dJJ_BSM')
plt.savefig('dJJ_SM_BSM.png')


L_HH  = BSM_model.predict(LM_HH)
L_ttH = BSM_model.predict(LM_ttH)
L_ZH  = BSM_model.predict(LM_ZH)
L_JJ  = BSM_model.predict(LM_JJ)
L_HH_6= BSM_model.predict(LM_HH_6)

L_Data = BSM_model.predict(LM_Data)

H_HH  = SM_model.predict(HM_HH)
H_ttH = SM_model.predict(HM_ttH)
H_ZH  = SM_model.predict(HM_ZH)
H_JJ  = SM_model.predict(HM_JJ)
H_HH_6= SM_model.predict(HM_HH_6)

H_Data = SM_model.predict(HM_Data)

plt.figure()

plt.hist(H_Data[:,0], bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', color = 'black',density='True',linewidth=1.5)
plt.hist(H_HH[:,0],   bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step',  weights=HM_W_HH,density='True',linewidth=1.5)
plt.hist(H_HH_6[:,0], bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=HM_W_HH_6,density='True',linewidth=1.5)
plt.hist(H_ttH[:,0],  bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=HM_W_ttH,density='True',linewidth=1.5)
plt.hist(H_ZH[:,0],   bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=HM_W_ZH,density='True',linewidth=1.5)
plt.hist(H_JJ[:,0],   bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=HM_W_JJ,density='True',linewidth=1.5)

plt.yscale('log')
plt.xlabel('pHH')
plt.ylabel('Event / 0.02')
plt.title('SM Model, High mHH')
plt.legend(['Data Sideband','SM@NLO HH','BSM@NLO HH k=10','ttH','ZH','JJ'], loc='lower left')
plt.savefig('pHH_SM_Model_High_Mass_WithData.png')


plt.figure()
plt.hist(L_Data[:,0], bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', color = 'black', density='True',linewidth=1.5)
plt.hist(L_HH[:,0],   bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step',  weights=LM_W_HH,density='True',linewidth=1.5)
plt.hist(L_HH_6[:,0], bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=LM_W_HH_6,density='True',linewidth=1.5)
plt.hist(L_ttH[:,0],  bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=LM_W_ttH,density='True',linewidth=1.5)
plt.hist(L_ZH[:,0],   bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=LM_W_ZH,density='True',linewidth=1.5)
plt.hist(L_JJ[:,0],   bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=LM_W_JJ,density='True',linewidth=1.5)

plt.yscale('log')
plt.xlabel('pHH')
plt.ylabel('Event / 0.02')
plt.title('BSM Model, Low mHH')
plt.legend(['Data Sideband','SM@NLO HH','BSM@NLO HH k=10','ttH','ZH','JJ'], loc='lower left')
plt.savefig('pHH_BSM_Model_Low_Mass_WithData.png')



plt.figure()

plt.hist(H_Data[:,1], bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', color = 'black',density='True',linewidth=1.5)
plt.hist(H_HH[:,1],   bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step',  weights=HM_W_HH,density='True',linewidth=1.5)
plt.hist(H_HH_6[:,1], bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=HM_W_HH_6,density='True',linewidth=1.5)
plt.hist(H_ttH[:,1],  bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=HM_W_ttH,density='True',linewidth=1.5)
plt.hist(H_ZH[:,1],   bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=HM_W_ZH,density='True',linewidth=1.5)
plt.hist(H_JJ[:,1],   bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=HM_W_JJ,density='True',linewidth=1.5)

plt.yscale('log')
plt.xlabel('pttH')
plt.ylabel('Event / 0.02')
plt.title('SM Model, High mHH')
plt.legend(['Data Sideband','SM@NLO HH','BSM@NLO HH k=10','ttH','ZH','JJ'], loc='lower left')
plt.savefig('pttH_SM_Model_High_Mass_WithData.png')


plt.figure()
plt.hist(L_Data[:,1], bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', color = 'black',density='True',linewidth=1.5)
plt.hist(L_HH[:,1],   bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step',  weights=LM_W_HH,density='True',linewidth=1.5)
plt.hist(L_HH_6[:,1], bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=LM_W_HH_6,density='True',linewidth=1.5)
plt.hist(L_ttH[:,1],  bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=LM_W_ttH,density='True',linewidth=1.5)
plt.hist(L_ZH[:,1],   bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=LM_W_ZH,density='True',linewidth=1.5)
plt.hist(L_JJ[:,1],   bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=LM_W_JJ,density='True',linewidth=1.5)

plt.yscale('log')
plt.xlabel('pttH')
plt.ylabel('Event / 0.02')
plt.title('BSM Model, Low mHH')
plt.legend(['Data Sideband','SM@NLO HH','BSM@NLO HH k=10','ttH','ZH','JJ'], loc='lower left')
plt.savefig('pttH_BSM_Model_Low_Mass_WithData.png')



plt.figure()

plt.hist(H_Data[:,2], bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', color = 'black',density='True',linewidth=1.5)
plt.hist(H_HH[:,2],   bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step',  weights=HM_W_HH,density='True',linewidth=1.5)
plt.hist(H_HH_6[:,2], bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=HM_W_HH_6,density='True',linewidth=1.5)
plt.hist(H_ttH[:,2],  bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=HM_W_ttH,density='True',linewidth=1.5)
plt.hist(H_ZH[:,2],   bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=HM_W_ZH,density='True',linewidth=1.5)
plt.hist(H_JJ[:,2],   bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=HM_W_JJ,density='True',linewidth=1.5)

plt.yscale('log')
plt.xlabel('pZH')
plt.ylabel('Event / 0.02')
plt.title('SM Model, High mHH')
plt.legend(['Data Sideband','SM@NLO HH','BSM@NLO HH k=10','ttH','ZH','JJ'], loc='lower left')
plt.savefig('pZH_SM_Model_High_Mass_WithData.png')


plt.figure()
plt.hist(L_Data[:,2], bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', color = 'black',density='True',linewidth=1.5)
plt.hist(L_HH[:,2],   bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step',  weights=LM_W_HH,density='True',linewidth=1.5)
plt.hist(L_HH_6[:,2], bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=LM_W_HH_6,density='True',linewidth=1.5)
plt.hist(L_ttH[:,2],  bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=LM_W_ttH,density='True',linewidth=1.5)
plt.hist(L_ZH[:,2],   bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=LM_W_ZH,density='True',linewidth=1.5)
plt.hist(L_JJ[:,2],   bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=LM_W_JJ,density='True',linewidth=1.5)

plt.yscale('log')
plt.xlabel('pZH')
plt.ylabel('Event / 0.02')
plt.title('BSM Model, Low mHH')
plt.legend(['Data Sideband','SM@NLO HH','BSM@NLO HH k=10','ttH','ZH','JJ'], loc='lower left')
plt.savefig('pZH_BSM_Model_Low_Mass_WithData.png')



plt.figure()

plt.hist(H_Data[:,3], bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', color = 'black',density='True',linewidth=1.5)
plt.hist(H_HH[:,3],   bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step',  weights=HM_W_HH,density='True',linewidth=1.5)
plt.hist(H_HH_6[:,3], bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=HM_W_HH_6,density='True',linewidth=1.5)
plt.hist(H_ttH[:,3],  bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=HM_W_ttH,density='True',linewidth=1.5)
plt.hist(H_ZH[:,3],   bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=HM_W_ZH,density='True',linewidth=1.5)
plt.hist(H_JJ[:,3],   bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=HM_W_JJ,density='True',linewidth=1.5)

plt.yscale('log')
plt.xlabel('pJJ')
plt.ylabel('Event / 0.02')
plt.title('SM Model, High mHH')
plt.legend(['Data Sideband','SM@NLO HH','BSM@NLO HH k=10','ttH','ZH','JJ'], loc='lower left')
plt.savefig('pJJ_SM_Model_High_Mass_WithData.png')


plt.figure()
plt.hist(L_Data[:,3], bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', color = 'black',density='True',linewidth=1.5)
plt.hist(L_HH[:,3],   bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step',  weights=LM_W_HH,density='True',linewidth=1.5)
plt.hist(L_HH_6[:,3], bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=LM_W_HH_6,density='True',linewidth=1.5)
plt.hist(L_ttH[:,3],  bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=LM_W_ttH,density='True',linewidth=1.5)
plt.hist(L_ZH[:,3],   bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=LM_W_ZH,density='True',linewidth=1.5)
plt.hist(L_JJ[:,3],   bins=[0,0.2,0.4,0.6,0.8,0.9,1.], histtype='step', weights=LM_W_JJ,density='True',linewidth=1.5)

plt.yscale('log')
plt.xlabel('pJJ')
plt.ylabel('Event / 0.02')
plt.title('BSM Model, Low mHH')
plt.legend(['Data Sideband','SM@NLO HH','BSM@NLO HH k=10','ttH','ZH','JJ'], loc='lower left')
plt.savefig('pJJ_BSM_Model_Low_Mass_WithData.png')


dHH_HM  = DHH(H_HH[:,0], H_HH[:,1], H_HH[:,2], H_HH[:,3], 'SM')
dttH_HM = DHH(H_ttH[:,0], H_ttH[:,1], H_ttH[:,2], H_ttH[:,3], 'SM')
dZH_HM  = DHH(H_ZH[:,0], H_ZH[:,1], H_ZH[:,2], H_ZH[:,3], 'SM')
dJJ_HM  = DHH(H_JJ[:,0], H_JJ[:,1], H_JJ[:,2], H_JJ[:,3], 'SM')
dHH_6_HM= DHH(H_HH_6[:,0], H_HH_6[:,1], H_HH_6[:,2], H_HH_6[:,3], 'SM')
dData_HM= DHH(H_Data[:,0], H_Data[:,1], H_Data[:,2], H_Data[:,3], 'SM')

dHH_LM  = DHH(L_HH[:,0],  L_HH[:,1],  L_HH[:,2],  L_HH[:,3], 'BSM')
dttH_LM = DHH(L_ttH[:,0], L_ttH[:,1], L_ttH[:,2], L_ttH[:,3], 'BSM')
dZH_LM  = DHH(L_ZH[:,0],  L_ZH[:,1],  L_ZH[:,2],  L_ZH[:,3], 'BSM')
dJJ_LM  = DHH(L_JJ[:,0],  L_JJ[:,1],  L_JJ[:,2],  L_JJ[:,3], 'BSM')
dHH_6_LM= DHH(L_HH_6[:,0],L_HH_6[:,1],L_HH_6[:,2],L_HH_6[:,3], 'BSM')
dData_LM= DHH(L_Data[:,0],L_Data[:,1],L_Data[:,2],L_Data[:,3], 'BSM')

plt.figure()

plt.hist(dData_HM, bins=range(-20,-5), histtype='step', color = 'black',   density='True', linewidth=1.5)
plt.hist(dHH_HM,   bins=range(-20,-5), histtype='step', weights=HM_W_HH,   density='True', linewidth=1.5)
plt.hist(dHH_6_HM, bins=range(-20,-5), histtype='step', weights=HM_W_HH_6, density='True', linewidth=1.5)
plt.hist(dttH_HM,  bins=range(-20,-5), histtype='step', weights=HM_W_ttH,  density='True', linewidth=1.5)
plt.hist(dZH_HM,   bins=range(-20,-5), histtype='step', weights=HM_W_ZH,   density='True', linewidth=1.5)
plt.hist(dJJ_HM,   bins=range(-20,-5), histtype='step', weights=HM_W_JJ,   density='True', linewidth=1.5)

plt.yscale('log')
plt.xlabel('dHH_SM')
plt.ylabel('Event')
plt.title('SM Model, High mHH')
plt.legend(['Data Sideband','SM@NLO HH','BSM@NLO HH k=10','ttH','ZH','JJ'], loc='lower left')
plt.savefig('dHH_SM_Model_High_Mass_WithData.png')


plt.figure()
plt.hist(dData_LM, bins=range(-20,-5), histtype='step', color = 'black',   density='True', linewidth=1.5)
plt.hist(dHH_LM,   bins=range(-20,-5), histtype='step', weights=LM_W_HH,   density='True', linewidth=1.5)
plt.hist(dHH_6_LM, bins=range(-20,-5), histtype='step', weights=LM_W_HH_6, density='True', linewidth=1.5)
plt.hist(dttH_LM,  bins=range(-20,-5), histtype='step', weights=LM_W_ttH,  density='True', linewidth=1.5)
plt.hist(dZH_LM,   bins=range(-20,-5), histtype='step', weights=LM_W_ZH,   density='True', linewidth=1.5)
plt.hist(dJJ_LM,   bins=range(-20,-5), histtype='step', weights=LM_W_JJ,   density='True', linewidth=1.5)

plt.yscale('log')
plt.xlabel('dHH_BSM')
plt.ylabel('Event')
plt.title('BSM Model, Low mHH')
plt.legend(['Data Sideband','SM@NLO HH','BSM@NLO HH k=10','ttH','ZH','JJ'], loc='lower left')
plt.savefig('dHH_BSM_Model_Low_Mass_WithData.png')



mask_dHH = np.asarray(dHH_HM) > -11.63
mask_dZH = np.asarray(dZH_HM) > -11.63
mask_dttH= np.asarray(dZH_HM) > -11.63
mask_dJJ = np.asarray(dJJ_HM) > -11.63

X_B = HM_JJ
X_A = HM_JJ[mask_dJJ]

mask_eta_bb = abs(X_B[:,12]) < 2.5
mask_eta_yy = abs(X_B[:,15]) < 2.5
mask_pt_bb = X_B[:,14] > 1.2
mask_pt_yy = X_B[:,17] > 1.2

mask_eta = mask_eta_yy * mask_eta_bb
mask_pt  = mask_pt_yy 

var_name = ['b1.pt','b1.eta','b1.phi','b1.score','b2.pt','b2.eta','b2.phi','b2.score','y1.pt','y1.eta','y1.phi','y2.pt','y2.eta','y2.phi','bb.pt','bb.eta','bb.phi','yy.pt','yy.eta','yy.phi','met.TST','hh.pt']

#var_name = ['b1.pt','b1.eta','b1.phi','b1.score','b2.pt','b2.eta','b2.phi','b2.score','y1.pt','y1.eta','y1.phi','y2.pt','y2.eta','y2.phi','bb.eta','bb.phi','yy.eta','yy.phi','met.TST','hh.pt']

#var_name = ['b1.eta','b1.phi','b1.score','b2.eta','b2.phi','b2.score','y1.pt','y1.eta','y1.phi','y2.pt','y2.eta','y2.phi','bb.eta','bb.phi','yy.pt','yy.eta','yy.phi','met.TST','hh.pt']

for i in range(0,X_A.shape[1]-1):
	
	plt.figure()
	plt.hist(X_B[:,i], histtype='step', density='True', linewidth=1.5, bins=20)
	plt.hist(X_A[:,i], histtype='step', density='True', linewidth=1.5, bins=20)
	plt.legend(['Before Cut', 'After DNN Cut'])
	#plt.yscale('log')
	plt.xlabel('Variable')
	plt.ylabel('Event')
	plt.savefig('Var_JJ_new/var_'+str(i)+'.png')
	plt.close()

plt.figure()
plt.hist(JJ_B*1e-3, histtype='step', density='True', linewidth=1.5, bins=20, range=[60,200])
#plt.hist(JJ_B[mask_eta]*1e-3, histtype='step', density='True', linewidth=1.5, bins=50, range=[60,200], weights=HM_W_JJ[mask_eta])

plt.hist(JJ_B[mask_dJJ]*1e-3, histtype='step', density='True', linewidth=1.5, bins=20, range=[60,200])
plt.legend(['Before ut', 'After Cut'])
plt.title('DNN')
#plt.yscale('log')
plt.xlabel('Variable mbb')
plt.ylabel('Event')
plt.savefig('Var_JJ_new/var_mbb.png')
plt.close()


plt.figure()
plt.hist(JJ_Y*1e-3, histtype='step', density='True', linewidth=1.5, bins=20)
#plt.hist(JJ_Y[mask_eta]*1e-3, histtype='step', density='True', linewidth=1.5, bins=50, weights=HM_W_JJ[mask_eta])
plt.hist(JJ_Y[mask_dJJ]*1e-3, histtype='step', density='True', linewidth=1.5, bins=20)
plt.title('DNN')
plt.legend(['Before Cut', 'After Cut'])
#plt.yscale('log')
plt.xlabel('Variable myy')
plt.ylabel('Event')
plt.savefig('Var_JJ_new/var_myy.png')
plt.close()


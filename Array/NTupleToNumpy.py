#!/usr/bin/env python

import ROOT 
import uproot as pr
import pandas as pd
import numpy as np
from math import pi as PI
import math
from argparse import ArgumentParser
import sys

def getArgs():
	"""
	Get arguments from command line.
	"""
	args = ArgumentParser(description="Argumetns for NTupleToNumpy for ANN")
	args.add_argument('-s', '--samples', action='store', required=True, help='Samples name to process .txt splitted by ,')
	args.add_argument('-i', '--inputdir', action='store', default='/afs/cern.ch/work/m/mobelfki/ANN/NTuple/h025_77btag/output', help='Input Directory')
	args.add_argument('-f', '--features', action='store', required=True, help='Features list')
	args.add_argument('-o', '--outdir', action='store', default='h025_77btag/', help='Output Directory')
	return args.parse_args()

class NTupleToNumpy:
	"""
	Produce numpy arrays form NTuple class 
	The output is saved in Array/output directory
	Doesn't take any inputs 
	"""

	def __init__(self, name):
		self.name = name;
		print('NTupleToNumpy', name)
	
	def loadTree(self, path, tree):
		self.Tree = pr.open(path)[tree];
		
	def loadFeatures(self, txt):
		with open(txt,'r') as features_file:
			self.Features = features_file.read().splitlines();
		features_file.close();

	def setLabel(self, label):
		self.Label = label;

	def dPhi(self, ag1, ag2):
		
		dphi = ag2 - ag1 
		for i in range(0,dphi.shape[0]) :
			if dphi[i] > PI : 
				dphi[i] = dphi[i] - 2*PI
			if dphi[i] < -PI :
				dphi[i] = dphi[i] + 2*PI
		return dphi 

	def dR(self, eta1, eta2, phi1, phi2):

		deta = math.pow(eta2-eta1, 2)
		dphi = math.pow(phi2-phi1, 2)

		dr = math.sqrt( deta + dphi )
		
		return dr

	def Execute(self):
		
		df = pd.DataFrame(self.Tree.arrays(self.Features, namedecode="utf-8"));
		
		df = df[self.Features]
		
		df['b1.pt'] = df['b1.pt'] / df['bb.m'];
		df['b2.pt'] = df['b2.pt'] / df['bb.m'];
		df['bb.pt'] = df['bb.pt'] / df['bb.m'];
		
		df['y1.pt'] = df['y1.pt'] / df['yy.m'];
		df['y2.pt'] = df['y2.pt'] / df['yy.m'];
		df['yy.pt'] = df['yy.pt'] / df['yy.m'];
		df['hh.pt'] = df['hh.pt'] / df['hh.m'];

		df['y1.phi'] = self.dPhi(df['y1.phi'],df['y1.phi'])
		df['y2.phi'] = self.dPhi(df['y1.phi'],df['y2.phi'])

		df['b1.phi'] = self.dPhi(df['y1.phi'],df['b1.phi'])
		df['b2.phi'] = self.dPhi(df['y1.phi'],df['b2.phi'])

		df['bb.phi'] = self.dPhi(df['y1.phi'],df['bb.phi'])
		df['yy.phi'] = self.dPhi(df['y1.phi'],df['yy.phi'])

		"""
		df['bb.dr'] = self.dR(df['b1.eta'], df['b2.eta'], df['b1.phi'], df['b2.phi'])
		df['yy.dr'] = self.dR(df['y1.eta'], df['y2.eta'], df['y1.phi'], df['y2.phi'])

		df['y1bb.dr'] = self.dR(df['y1.eta'], df['bb.eta'], df['y1.phi'], df['bb.phi'])
		df['y2bb.dr'] = self.dR(df['y2.eta'], df['bb.eta'], df['y2.phi'], df['bb.phi'])

		df['b1yy.dr'] = self.dR(df['b1.eta'], df['yy.eta'], df['b1.phi'], df['yy.phi'])
		df['b2yy.dr'] = self.dR(df['b2.eta'], df['yy.eta'], df['b2.phi'], df['yy.phi'])
		"""

		#df['hh.m']  = df['hh.m'] - df['yy.m'] - df['bb.m'] + 250000;
		
		#sumW = df['Event.TotWeight'].sum()

		df['mx'] = df['hh.m'] - df['yy.m'] - df['bb.m'] + 250000;

		df['isHH']  = self.Label[0];
		df['isttH'] = self.Label[1];
		df['isZH']  = self.Label[2];
		df['isJJ']  = self.Label[3];

		df = df.drop(['hh.m'], axis=1)
		#df = df.drop(['y1.phi'], axis=1)
	
		return df;

def main():
	"""
	The main function of NTupleToNumpy
	"""
	args=getArgs()
	with open(args.samples,'r') as samples_file:
		samples_name = samples_file.read().splitlines();
	samples_file.close();
	features_file = args.features
	trees = {'Tree','Tree_Train','Tree_Val','Tree_Test'}
	for sample in samples_name:
		
		NTuple2Numpy = NTupleToNumpy(sample);
		for tree in trees:
			NTuple2Numpy.loadTree(args.inputdir+'/'+sample,tree);
			NTuple2Numpy.loadFeatures(features_file);
			label = np.array([-1,-1,-1,-1]);
			
			if "aMCnloHwpp_hh_yybb_AF2" in sample:
				label = np.array([1,0,0,0]);
			if "MGPy8_hh_yybb_" in sample:
				label = np.array([1,0,0,0]);
			if "PowhegH7_HHbbyy_" in sample:
				label = np.array([1,0,0,0]);
			if "ZH125J" in sample:
				label = np.array([0,0,1,0]);
			if "ggZH125" in sample:
				label = np.array([0,0,1,0]);	
			if "ttH125" in sample:
				label = np.array([0,1,0,0]);
			if "aMCnloPy8_tWH125" in sample:
				label = np.array([0,1,0,0]);
			if "Sherpa2_diphoton" in sample:
				label = np.array([0,0,0,1]);
			if "Sherpa2_diphoton" in sample:
				label = np.array([0,0,0,1]);
			if "PowhegPy8_NNLOPS_ggH125" in sample:
				label = np.array([0,0,0,1]);
			if "PowhegPy8_bbH125" in sample:
				label = np.array([0,0,0,1]);
			if "DataFullRun2" in sample:
				label = np.array([0,0,0,0]);	
			if label.all() == np.array([-1,-1,-1,-1]).all():
				print("The sample that you give is not setted --check the code--", sample)
				quit()	
		
			NTuple2Numpy.setLabel(label);

			print(NTuple2Numpy.Label)
	
			out = NTuple2Numpy.Execute();
		
			np.save(args.outdir+'/'+sample+'.'+tree+'.npy', out.to_numpy())
		
if __name__ == '__main__':
	main()
		

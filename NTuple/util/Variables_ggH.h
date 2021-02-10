#include <memory>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>

// ROOT include(s):
#include <TFile.h>
#include <TH1F.h>
#include <TChain.h>
#include <TError.h>
#include <TString.h>
#include <TStopwatch.h>
#include <TSystem.h>
#include "TLorentzVector.h"
#include "TRandom3.h"
bool _isData = false;
bool _isDirectory = false;
Int_t _nSample = 1;
Int_t _nCompain = 3;
TString _GlobalPath = "/eos/user/m/mobelfki/yybb_Samples/"; 
TString _Compain [] = {"mc16a","mc16d","mc16e"};
TString _Tag[] = {"e5607_s3126_r9364_p4097","e5607_s3126_r10201_p4097","e5607_s3126_r10724_p4097"};
TString _Type = ".MxAODDetailed";
TString _Version = "_h025";
TString _Samplename[]  = {".PowhegPy8_NNLOPS_ggH125"};
TString _histo_name = "CutFlow_PowhegPy8_NNLOPS_ggH125_noDalitz_weighted";

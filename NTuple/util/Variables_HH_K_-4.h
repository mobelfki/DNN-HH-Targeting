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
TString _GlobalPath = "/eos/atlas/atlascerngroupdisk/phys-higgs/HSG1/MxAOD/h025/"; 
TString _Compain [] = {"mc16a","mc16d","mc16e"};
TString _Tag[] = {"e5504_a875_r9364_p4097","e5504_a875_r10201_p4097","e5504_a875_r10724_p4097"};
TString _Type = ".MxAODDetailedNoSkim";
TString _Version = "_h025";
TString _Samplename[]  = {".MGPy8_hh_yybb_minus_lambda04_AF2"};
TString _histo_name = "CutFlow_MGPy8_hh_yybb_minus_lambda04_noDalitz_weighted";

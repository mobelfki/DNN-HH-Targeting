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

bool _isDirectory = false;
bool _isData = false;
Int_t _nSample = 1;
Int_t _nCompain = 3;
TString _GlobalPath = "/eos/atlas/atlascerngroupdisk/phys-higgs/HSG1/MxAOD/h025/"; 
TString _Compain [] = {"mc16a","mc16d","mc16e"};
TString _Tag[] = {"e6050_s3126_r9364_p4097","e6050_s3126_r10201_p4097","e6050_s3126_r10724_p4097"};
TString _Type = ".MxAODDetailed";
TString _Version = "_h025";
TString _Samplename[]  = {".PowhegPy8_bbH125"};
TString _histo_name = "CutFlow_PowhegPy8_bbH125_noDalitz_weighted";


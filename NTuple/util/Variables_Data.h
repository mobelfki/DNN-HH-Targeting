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

bool _isData = true;
bool _isDirectory = false;
Int_t _nSample = 1;
Int_t _nCompain = 3;
TString _GlobalPath = "/eos/user/m/mobelfki/yybb_Samples/"; 
TString _Compain [] = {"data15","data17","data18"};
TString _Tag[] = {"","",""};
TString _Type = "";
TString _Version = "";
TString _Samplename[]  = {""};
TString _histo_name = "CutFlow_Run351698";


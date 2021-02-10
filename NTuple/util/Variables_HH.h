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
TString _Tag[] = {"e4419_a875_r9364_p4097","e4419_a875_r10201_p4097","e4419_a875_r10724_p4097"};
TString _Type = ".MxAODDetailedNoSkim";
TString _Version = "_h025";
TString _Samplename[]  = {".aMCnloHwpp_hh_yybb_AF2"};
TString _histo_name = "CutFlow_aMCnlo_Hwpp_hh_yybb_noDalitz_weighted";

//mc16a.aMCnloHwpp_hh_yybb_AF2.MxAODDetailedNoSkim.e4419_a875_r9364_p4097_h025.root
//mc16d.aMCnloHwpp_hh_yybb_AF2.MxAODDetailedNoSkim.e4419_a875_r10201_p4097_h025.root
//mc16e.aMCnloHwpp_hh_yybb_AF2.MxAODDetailedNoSkim.e4419_a875_r10724_p4097_h025.root


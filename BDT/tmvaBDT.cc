#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <TFile.h>
#include <TString.h>

#include <TMVA/DataLoader.h>
#include <TMVA/Factory.h>

int tmvaBDT() {  

// Create ouput file, factory object and open the input file

  TString outfileName( "BDT_Inclusive_SM_NFM_M.root" );
  TFile* outputFile = TFile::Open( outfileName, "RECREATE" );
  
  TMVA::Factory* factory = new TMVA::Factory( "TMVAMulticlass", outputFile, "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Multiclass" );
	
  TMVA::DataLoader* dataloader = new TMVA::DataLoader("BDT_Inclusive_SM_NFM_M");
  
  TFile* input_S = new TFile("/afs/cern.ch/work/m/mobelfki/ANN/NTuple/output/output.aMCnloHwpp_hh_yybb_AF2.MxAODDetailed.h024.root" ,"READ");
 // TFile* input_S = new TFile("/afs/cern.ch/work/m/mobelfki/ANN/NTuple/output/output.MGPy8_hh_yybb_plus_lambda06_AF2.MxAODDetailed.h024.root","READ");
  TFile* input_B1 = new TFile("/afs/cern.ch/work/m/mobelfki/ANN/NTuple/output/output.PowhegPy8_ttH125.MxAODDetailed.h024.root","READ");
  TFile* input_B2 = new TFile("/afs/cern.ch/work/m/mobelfki/ANN/NTuple/output/output.PowhegPy8_ZH125J.MxAODDetailed.h024.root","READ");
  TFile* input_B3 = new TFile("/afs/cern.ch/work/m/mobelfki/ANN/NTuple/output/output.Sherpa2_diphoton_myy_90_175_AF2.MxAODDetailed.h024.root","READ");
		
  TTree *signalTree_Train       = (TTree*)input_S->Get("Tree_Train");
  TTree *background_1_Train     = (TTree*)input_B1->Get("Tree_Train");
  TTree *background_2_Train     = (TTree*)input_B2->Get("Tree_Train");
  TTree *background_3_Train     = (TTree*)input_B3->Get("Tree_Train");

  TTree *signalTree_Test       = (TTree*)input_S->Get("Tree_Test");
  TTree *background_1_Test     = (TTree*)input_B1->Get("Tree_Test");
  TTree *background_2_Test     = (TTree*)input_B2->Get("Tree_Test");
  TTree *background_3_Test     = (TTree*)input_B3->Get("Tree_Test");


  double sigWeight = 1.0;
  double bkgWeight = 1.0;

  TCut mycuts = "(hh.m - yy.m - bb.m + 250000)*1e-3 >= 0"; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
  TCut mycutb = "(hh.m - yy.m - bb.m + 250000)*1e-3 >= 0"; // for example: TCut mycutb = "abs(var1)<0.5";

  dataloader->AddTree(signalTree_Train, "Signal_Train", sigWeight, "", TMVA::Types::kTraining );
  dataloader->AddTree (background_1_Train, "ttH_Train", bkgWeight, "", TMVA::Types::kTraining);
  dataloader->AddTree (background_2_Train, "ZH_Train", bkgWeight,  "", TMVA::Types::kTraining);
  dataloader->AddTree (background_3_Train, "JJ_Train", bkgWeight,  "", TMVA::Types::kTraining);

  dataloader->AddTree(signalTree_Test, "Signal_Test", sigWeight, "", TMVA::Types::kTesting);
  dataloader->AddTree (background_1_Test, "ttH_Test", bkgWeight, "",  TMVA::Types::kTesting);
  dataloader->AddTree (background_2_Test, "ZH_Test", bkgWeight,  "",  TMVA::Types::kTesting);
  dataloader->AddTree (background_3_Test, "JJ_Test", bkgWeight,  "",  TMVA::Types::kTesting);


//  dataloader->SetSignalWeightExpression("Event.TotWeight");
//  dataloader->SetBackgroundWeightExpression("Event.TotWeight");

dataloader->AddVariable( "b1.eta", 'F' );
dataloader->AddVariable( "b2.eta", 'F' );
dataloader->AddVariable( "y1.eta", 'F' );
dataloader->AddVariable( "y2.eta", 'F' );
dataloader->AddVariable( "bb.eta", 'F' );
dataloader->AddVariable( "yy.eta", 'F' );

dataloader->AddVariable( "b1.phi", 'F' );
dataloader->AddVariable( "b2.phi", 'F' );
dataloader->AddVariable( "y1.phi", 'F' );
dataloader->AddVariable( "y2.phi", 'F' );
dataloader->AddVariable( "bb.phi", 'F' );
dataloader->AddVariable( "yy.phi", 'F' );

dataloader->AddVariable( "b1.pt := b1.pt/bb.m", 'F' );
dataloader->AddVariable( "b2.pt := b2.pt/bb.m", 'F' );
dataloader->AddVariable( "y1.pt := y1.pt/yy.m", 'F' );
dataloader->AddVariable( "y2.pt := y2.pt/yy.m", 'F' );
dataloader->AddVariable( "bb.pt := bb.pt/bb.m", 'F' );
dataloader->AddVariable( "yy.pt := yy.pt/yy.m", 'F' );

dataloader->AddVariable( "b1.score", 'F' );
dataloader->AddVariable( "b2.score", 'F' );

dataloader->AddVariable( "bb.dr", 'F' );
dataloader->AddVariable( "yy.dr", 'F' );
dataloader->AddVariable( "bb.m", 'F' );


dataloader->AddVariable( "bb.H1_l1 := 1", 'F' );
//dataloader->AddVariable( "bb.HY_l3", 'F' );
//dataloader->AddVariable( "yy.H1_l3", 'F' );
//dataloader->AddVariable( "yy.HP_l1", 'F' );
//dataloader->AddVariable( "hh.H1_l3", 'F' );
//dataloader->AddVariable( "hh.HT_l1", 'F' );
//dataloader->AddVariable( "hh.HP_l2", 'F' );


dataloader->AddSpectator( "yy.m", 'F' );
dataloader->AddSpectator( "X.m := (hh.m - yy.m - bb.m + 250000)", 'F' );
dataloader->AddSpectator( "Event.Cat", 'I' );
dataloader->AddSpectator( "Event.TotWeight", 'I' );

  dataloader->PrepareTrainingAndTestTree(mycuts, mycutb,"NormMode=None" );


     factory->BookMethod( dataloader, TMVA::Types::kBDT, "BDTG",
			  "!H:!V:NTrees=1000:MinNodeSize=2.5%:BoostType=Grad:Shrinkage=0.10:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20:MaxDepth=2" );


// Train, test and evaluate all methods

   factory->TrainAllMethods();
   factory->TestAllMethods();
   factory->EvaluateAllMethods();    

// Save the output and finish up

  outputFile->Close();
  std::cout << "==> wrote root file TMVA.root" << std::endl;
  std::cout << "==> TMVAnalysis is done!" << std::endl; 

  delete factory;
	if (!gROOT->IsBatch()) TMVA::TMVAGui( outfileName );
  return 0;

}

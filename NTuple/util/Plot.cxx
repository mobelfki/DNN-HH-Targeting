#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TH1F.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include "vector"
#include "vector"
using namespace std;

void Plot(TChain* chain);
TChain* GetTChain(TString name);

// Declaration of leaf types
   Double_t        b1_pt;
   Double_t        b2_pt;
   Double_t        b1_eta;
   Double_t        b2_eta;
   Double_t        b1_phi;
   Double_t        b2_phi;
   Double_t        b1_e;
   Double_t        b2_e;
   Double_t        b1_score;
   Double_t        b2_score;
   Double_t        y1_pt;
   Double_t        y2_pt;
   Double_t        y1_eta;
   Double_t        y2_eta;
   Double_t        y1_phi;
   Double_t        y2_phi;
   Double_t        y1_e;
   Double_t        y2_e;
   Double_t        hh_pt;
   Double_t        yy_pt;
   Double_t        bb_pt;
   Double_t        hh_eta;
   Double_t        yy_eta;
   Double_t        bb_eta;
   Double_t        hh_phi;
   Double_t        yy_phi;
   Double_t        bb_phi;
   Double_t        hh_e;
   Double_t        yy_e;
   Double_t        bb_e;
   Double_t        hh_m;
   Double_t        yy_m;
   Double_t        bb_m;
   Double_t        hh_dr;
   Double_t        yy_dr;
   Double_t        bb_dr;
   Double_t        b1yy_dr;
   Double_t        b2yy_dr;
   Double_t        y1bb_dr;
   Double_t        y2bb_dr;
   Double_t        AddJ_pt;
   Double_t        AddJ_eta;
   Double_t        AddJ_phi;
   Double_t        AddJ_m;
   Int_t           AddJ_N;
   Int_t           Jet_N;
   Int_t           AddBJ_N;

   Double_t        yy_HT_l1;
   Double_t        yy_HT_l2;
   Double_t        yy_HT_l3;
   Double_t        yy_HT_l4;
   Double_t        yy_HT_l5;
   Double_t        yy_HP_l1;
   Double_t        yy_HP_l2;
   Double_t        yy_HP_l3;
   Double_t        yy_HP_l4;
   Double_t        yy_HP_l5;
   Double_t        yy_HZ_l1;
   Double_t        yy_HZ_l2;
   Double_t        yy_HZ_l3;
   Double_t        yy_HZ_l4;
   Double_t        yy_HZ_l5;
   Double_t        yy_HY_l1;
   Double_t        yy_HY_l2;
   Double_t        yy_HY_l3;
   Double_t        yy_HY_l4;
   Double_t        yy_HY_l5;
   Double_t        yy_HS_l1;
   Double_t        yy_HS_l2;
   Double_t        yy_HS_l3;
   Double_t        yy_HS_l4;
   Double_t        yy_HS_l5;
   Double_t        yy_H1_l1;
   Double_t        yy_H1_l2;
   Double_t        yy_H1_l3;
   Double_t        yy_H1_l4;
   Double_t        yy_H1_l5;
   Double_t        bb_HT_l1;
   Double_t        bb_HT_l2;
   Double_t        bb_HT_l3;
   Double_t        bb_HT_l4;
   Double_t        bb_HT_l5;
   Double_t        bb_HP_l1;
   Double_t        bb_HP_l2;
   Double_t        bb_HP_l3;
   Double_t        bb_HP_l4;
   Double_t        bb_HP_l5;
   Double_t        bb_HZ_l1;
   Double_t        bb_HZ_l2;
   Double_t        bb_HZ_l3;
   Double_t        bb_HZ_l4;
   Double_t        bb_HZ_l5;
   Double_t        bb_HY_l1;
   Double_t        bb_HY_l2;
   Double_t        bb_HY_l3;
   Double_t        bb_HY_l4;
   Double_t        bb_HY_l5;
   Double_t        bb_HS_l1;
   Double_t        bb_HS_l2;
   Double_t        bb_HS_l3;
   Double_t        bb_HS_l4;
   Double_t        bb_HS_l5;
   Double_t        bb_H1_l1;
   Double_t        bb_H1_l2;
   Double_t        bb_H1_l3;
   Double_t        bb_H1_l4;
   Double_t        bb_H1_l5;
   Double_t        yybb_HT_l1;
   Double_t        yybb_HT_l2;
   Double_t        yybb_HT_l3;
   Double_t        yybb_HT_l4;
   Double_t        yybb_HT_l5;
   Double_t        yybb_HP_l1;
   Double_t        yybb_HP_l2;
   Double_t        yybb_HP_l3;
   Double_t        yybb_HP_l4;
   Double_t        yybb_HP_l5;
   Double_t        yybb_HZ_l1;
   Double_t        yybb_HZ_l2;
   Double_t        yybb_HZ_l3;
   Double_t        yybb_HZ_l4;
   Double_t        yybb_HZ_l5;
   Double_t        yybb_HY_l1;
   Double_t        yybb_HY_l2;
   Double_t        yybb_HY_l3;
   Double_t        yybb_HY_l4;
   Double_t        yybb_HY_l5;
   Double_t        yybb_HS_l1;
   Double_t        yybb_HS_l2;
   Double_t        yybb_HS_l3;
   Double_t        yybb_HS_l4;
   Double_t        yybb_HS_l5;
   Double_t        yybb_H1_l1;
   Double_t        yybb_H1_l2;
   Double_t        yybb_H1_l3;
   Double_t        yybb_H1_l4;
   Double_t        yybb_H1_l5;
   Double_t        hh_HT_l1;
   Double_t        hh_HT_l2;
   Double_t        hh_HT_l3;
   Double_t        hh_HT_l4;
   Double_t        hh_HT_l5;
   Double_t        hh_HP_l1;
   Double_t        hh_HP_l2;
   Double_t        hh_HP_l3;
   Double_t        hh_HP_l4;
   Double_t        hh_HP_l5;
   Double_t        hh_HZ_l1;
   Double_t        hh_HZ_l2;
   Double_t        hh_HZ_l3;
   Double_t        hh_HZ_l4;
   Double_t        hh_HZ_l5;
   Double_t        hh_HY_l1;
   Double_t        hh_HY_l2;
   Double_t        hh_HY_l3;
   Double_t        hh_HY_l4;
   Double_t        hh_HY_l5;
   Double_t        hh_HS_l1;
   Double_t        hh_HS_l2;
   Double_t        hh_HS_l3;
   Double_t        hh_HS_l4;
   Double_t        hh_HS_l5;
   Double_t        hh_H1_l1;
   Double_t        hh_H1_l2;
   Double_t        hh_H1_l3;
   Double_t        hh_H1_l4;
   Double_t        hh_H1_l5;
   vector<double>  *HT;
   vector<double>  *HS;
   vector<double>  *HP;
   vector<double>  *H1;
   vector<double>  *HY;
   vector<double>  *L;
   vector<double>  *met_px;
   vector<double>  *met_py;
   vector<double>  *met_e;
   vector<string>  *met_name;
   Float_t         Event_TotWeight;
   ULong64_t       Event_Number;
   Float_t         Event_bbyyWeight;
   Float_t         Event_Weight;
   Float_t         Event_Xsec;
   Float_t         Event_Lumi;
   Float_t         Event_N;
   Int_t           Event_Cat;

   // List of branches
   TBranch        *b_b1_pt;   //!
   TBranch        *b_b2_pt;   //!
   TBranch        *b_b1_eta;   //!
   TBranch        *b_b2_eta;   //!
   TBranch        *b_b1_phi;   //!
   TBranch        *b_b2_phi;   //!
   TBranch        *b_b1_e;   //!
   TBranch        *b_b2_e;   //!
   TBranch        *b_b1_score;   //!
   TBranch        *b_b2_score;   //!
   TBranch        *b_y1_pt;   //!
   TBranch        *b_y2_pt;   //!
   TBranch        *b_y1_eta;   //!
   TBranch        *b_y2_eta;   //!
   TBranch        *b_y1_phi;   //!
   TBranch        *b_y2_phi;   //!
   TBranch        *b_y1_e;   //!
   TBranch        *b_y2_e;   //!
   TBranch        *b_hh_pt;   //!
   TBranch        *b_yy_pt;   //!
   TBranch        *b_bb_pt;   //!
   TBranch        *b_hh_eta;   //!
   TBranch        *b_yy_eta;   //!
   TBranch        *b_bb_eta;   //!
   TBranch        *b_hh_phi;   //!
   TBranch        *b_yy_phi;   //!
   TBranch        *b_bb_phi;   //!
   TBranch        *b_hh_e;   //!
   TBranch        *b_yy_e;   //!
   TBranch        *b_bb_e;   //!
   TBranch        *b_hh_m;   //!
   TBranch        *b_yy_m;   //!
   TBranch        *b_bb_m;   //!
   TBranch        *b_hh_dr;   //!
   TBranch        *b_yy_dr;   //!
   TBranch        *b_bb_dr;   //!
   TBranch        *b_b1yy_dr;   //!
   TBranch        *b_b2yy_dr;   //!
   TBranch        *b_y1bb_dr;   //!
   TBranch        *b_y2bb_dr;   //!
   TBranch        *b_AddJ_pt;   //!
   TBranch        *b_AddJ_eta;   //!
   TBranch        *b_AddJ_phi;   //!
   TBranch        *b_AddJ_m;   //!
   TBranch        *b_AddJ_N;   //!
   TBranch        *b_Jet_N;   //!
   TBranch        *b_AddBJ_N;   //!
   TBranch        *b_yy_HT_l1;   //!
   TBranch        *b_yy_HT_l2;   //!
   TBranch        *b_yy_HT_l3;   //!
   TBranch        *b_yy_HT_l4;   //!
   TBranch        *b_yy_HT_l5;   //!
   TBranch        *b_yy_HP_l1;   //!
   TBranch        *b_yy_HP_l2;   //!
   TBranch        *b_yy_HP_l3;   //!
   TBranch        *b_yy_HP_l4;   //!
   TBranch        *b_yy_HP_l5;   //!
   TBranch        *b_yy_HZ_l1;   //!
   TBranch        *b_yy_HZ_l2;   //!
   TBranch        *b_yy_HZ_l3;   //!
   TBranch        *b_yy_HZ_l4;   //!
   TBranch        *b_yy_HZ_l5;   //!
   TBranch        *b_yy_HY_l1;   //!
   TBranch        *b_yy_HY_l2;   //!
   TBranch        *b_yy_HY_l3;   //!
   TBranch        *b_yy_HY_l4;   //!
   TBranch        *b_yy_HY_l5;   //!
   TBranch        *b_yy_HS_l1;   //!
   TBranch        *b_yy_HS_l2;   //!
   TBranch        *b_yy_HS_l3;   //!
   TBranch        *b_yy_HS_l4;   //!
   TBranch        *b_yy_HS_l5;   //!
   TBranch        *b_yy_H1_l1;   //!
   TBranch        *b_yy_H1_l2;   //!
   TBranch        *b_yy_H1_l3;   //!
   TBranch        *b_yy_H1_l4;   //!
   TBranch        *b_yy_H1_l5;   //!
   TBranch        *b_bb_HT_l1;   //!
   TBranch        *b_bb_HT_l2;   //!
   TBranch        *b_bb_HT_l3;   //!
   TBranch        *b_bb_HT_l4;   //!
   TBranch        *b_bb_HT_l5;   //!
   TBranch        *b_bb_HP_l1;   //!
   TBranch        *b_bb_HP_l2;   //!
   TBranch        *b_bb_HP_l3;   //!
   TBranch        *b_bb_HP_l4;   //!
   TBranch        *b_bb_HP_l5;   //!
   TBranch        *b_bb_HZ_l1;   //!
   TBranch        *b_bb_HZ_l2;   //!
   TBranch        *b_bb_HZ_l3;   //!
   TBranch        *b_bb_HZ_l4;   //!
   TBranch        *b_bb_HZ_l5;   //!
   TBranch        *b_bb_HY_l1;   //!
   TBranch        *b_bb_HY_l2;   //!
   TBranch        *b_bb_HY_l3;   //!
   TBranch        *b_bb_HY_l4;   //!
   TBranch        *b_bb_HY_l5;   //!
   TBranch        *b_bb_HS_l1;   //!
   TBranch        *b_bb_HS_l2;   //!
   TBranch        *b_bb_HS_l3;   //!
   TBranch        *b_bb_HS_l4;   //!
   TBranch        *b_bb_HS_l5;   //!
   TBranch        *b_bb_H1_l1;   //!
   TBranch        *b_bb_H1_l2;   //!
   TBranch        *b_bb_H1_l3;   //!
   TBranch        *b_bb_H1_l4;   //!
   TBranch        *b_bb_H1_l5;   //!
   TBranch        *b_yybb_HT_l1;   //!
   TBranch        *b_yybb_HT_l2;   //!
   TBranch        *b_yybb_HT_l3;   //!
   TBranch        *b_yybb_HT_l4;   //!
   TBranch        *b_yybb_HT_l5;   //!
   TBranch        *b_yybb_HP_l1;   //!
   TBranch        *b_yybb_HP_l2;   //!
   TBranch        *b_yybb_HP_l3;   //!
   TBranch        *b_yybb_HP_l4;   //!
   TBranch        *b_yybb_HP_l5;   //!
   TBranch        *b_yybb_HZ_l1;   //!
   TBranch        *b_yybb_HZ_l2;   //!
   TBranch        *b_yybb_HZ_l3;   //!
   TBranch        *b_yybb_HZ_l4;   //!
   TBranch        *b_yybb_HZ_l5;   //!
   TBranch        *b_yybb_HY_l1;   //!
   TBranch        *b_yybb_HY_l2;   //!
   TBranch        *b_yybb_HY_l3;   //!
   TBranch        *b_yybb_HY_l4;   //!
   TBranch        *b_yybb_HY_l5;   //!
   TBranch        *b_yybb_HS_l1;   //!
   TBranch        *b_yybb_HS_l2;   //!
   TBranch        *b_yybb_HS_l3;   //!
   TBranch        *b_yybb_HS_l4;   //!
   TBranch        *b_yybb_HS_l5;   //!
   TBranch        *b_yybb_H1_l1;   //!
   TBranch        *b_yybb_H1_l2;   //!
   TBranch        *b_yybb_H1_l3;   //!
   TBranch        *b_yybb_H1_l4;   //!
   TBranch        *b_yybb_H1_l5;   //!
   TBranch        *b_hh_HT_l1;   //!
   TBranch        *b_hh_HT_l2;   //!
   TBranch        *b_hh_HT_l3;   //!
   TBranch        *b_hh_HT_l4;   //!
   TBranch        *b_hh_HT_l5;   //!
   TBranch        *b_hh_HP_l1;   //!
   TBranch        *b_hh_HP_l2;   //!
   TBranch        *b_hh_HP_l3;   //!
   TBranch        *b_hh_HP_l4;   //!
   TBranch        *b_hh_HP_l5;   //!
   TBranch        *b_hh_HZ_l1;   //!
   TBranch        *b_hh_HZ_l2;   //!
   TBranch        *b_hh_HZ_l3;   //!
   TBranch        *b_hh_HZ_l4;   //!
   TBranch        *b_hh_HZ_l5;   //!
   TBranch        *b_hh_HY_l1;   //!
   TBranch        *b_hh_HY_l2;   //!
   TBranch        *b_hh_HY_l3;   //!
   TBranch        *b_hh_HY_l4;   //!
   TBranch        *b_hh_HY_l5;   //!
   TBranch        *b_hh_HS_l1;   //!
   TBranch        *b_hh_HS_l2;   //!
   TBranch        *b_hh_HS_l3;   //!
   TBranch        *b_hh_HS_l4;   //!
   TBranch        *b_hh_HS_l5;   //!
   TBranch        *b_hh_H1_l1;   //!
   TBranch        *b_hh_H1_l2;   //!
   TBranch        *b_hh_H1_l3;   //!
   TBranch        *b_hh_H1_l4;   //!
   TBranch        *b_hh_H1_l5;   //!
   TBranch        *b_HT;   //!
   TBranch        *b_HS;   //!
   TBranch        *b_HP;   //!
   TBranch        *b_H1;   //!
   TBranch        *b_HY;   //!
   TBranch        *b_L;   //!
   TBranch        *b_met_px;   //!
   TBranch        *b_met_py;   //!
   TBranch        *b_met_e;   //!
   TBranch        *b_met_name;   //!
   TBranch        *b_Event_TotWeight;   //!
   TBranch        *b_Event_Number;   //!
   TBranch        *b_Event_bbyyWeight;   //!
   TBranch        *b_Event_Weight;   //!
   TBranch        *b_Event_Xsec;   //!
   TBranch        *b_Event_Lumi;   //!
   TBranch        *b_Event_N;   //!
   TBranch        *b_Event_Cat;   //!

   TH1F* histo_MYY;
   TH1F* histo_MBB;
   TH1F* histo_AddPt;
   vector<TH1F*> histos;
   
   TH1F* histo_HT[19];
   TH1F* histo_HS[19];
   TH1F* histo_HP[19];
   TH1F* histo_H1[19];
   TH1F* histo_HY[19];
   TH1F* histo_L[6];

	TH1F* histo_bb_H1_l1;
	TH1F* histo_bb_HY_l3;
	TH1F* histo_hh_H1_l3;
	TH1F* histo_hh_HT_l1;
	TH1F* histo_hh_HP_l2;
	TH1F* histo_yy_HP_l1;
	TH1F* histo_yy_H1_l3;

TChain* GetTChain(TString name)
{


	   TFile* _input = new TFile(name,"READ");
	
	   TChain* fChain = (TChain*)_input->Get("Tree");
	
	   fChain->SetBranchAddress("b1.pt", &b1_pt, &b_b1_pt);
   	   fChain->SetBranchAddress("b2.pt", &b2_pt, &b_b2_pt);
   	   fChain->SetBranchAddress("b1.eta", &b1_eta, &b_b1_eta);
   	   fChain->SetBranchAddress("b2.eta", &b2_eta, &b_b2_eta);
   	   fChain->SetBranchAddress("b1.phi", &b1_phi, &b_b1_phi);
   	   fChain->SetBranchAddress("b2.phi", &b2_phi, &b_b2_phi);
   	   fChain->SetBranchAddress("b1.e", &b1_e, &b_b1_e);
   	   fChain->SetBranchAddress("b2.e", &b2_e, &b_b2_e);
   	   fChain->SetBranchAddress("b1.score", &b1_score, &b_b1_score);
   	   fChain->SetBranchAddress("b2.score", &b2_score, &b_b2_score);
   	   fChain->SetBranchAddress("y1.pt", &y1_pt, &b_y1_pt);
   	   fChain->SetBranchAddress("y2.pt", &y2_pt, &b_y2_pt);
	   fChain->SetBranchAddress("y1.eta", &y1_eta, &b_y1_eta);
	   fChain->SetBranchAddress("y2.eta", &y2_eta, &b_y2_eta);
	   fChain->SetBranchAddress("y1.phi", &y1_phi, &b_y1_phi);
	   fChain->SetBranchAddress("y2.phi", &y2_phi, &b_y2_phi);
	   fChain->SetBranchAddress("y1.e", &y1_e, &b_y1_e);
	   fChain->SetBranchAddress("y2.e", &y2_e, &b_y2_e);
	   fChain->SetBranchAddress("hh.pt", &hh_pt, &b_hh_pt);
	   fChain->SetBranchAddress("yy.pt", &yy_pt, &b_yy_pt);
	   fChain->SetBranchAddress("bb.pt", &bb_pt, &b_bb_pt);
	   fChain->SetBranchAddress("hh.eta", &hh_eta, &b_hh_eta);
	   fChain->SetBranchAddress("yy.eta", &yy_eta, &b_yy_eta);
	   fChain->SetBranchAddress("bb.eta", &bb_eta, &b_bb_eta);
	   fChain->SetBranchAddress("hh.phi", &hh_phi, &b_hh_phi);
	   fChain->SetBranchAddress("yy.phi", &yy_phi, &b_yy_phi);
	   fChain->SetBranchAddress("bb.phi", &bb_phi, &b_bb_phi);
	   fChain->SetBranchAddress("hh.e", &hh_e, &b_hh_e);
	   fChain->SetBranchAddress("yy.e", &yy_e, &b_yy_e);
	   fChain->SetBranchAddress("bb.e", &bb_e, &b_bb_e);
	   fChain->SetBranchAddress("hh.m", &hh_m, &b_hh_m);
	   fChain->SetBranchAddress("yy.m", &yy_m, &b_yy_m);
	   fChain->SetBranchAddress("bb.m", &bb_m, &b_bb_m);
	   fChain->SetBranchAddress("hh.dr", &hh_dr, &b_hh_dr);
	   fChain->SetBranchAddress("yy.dr", &yy_dr, &b_yy_dr);
	   fChain->SetBranchAddress("bb.dr", &bb_dr, &b_bb_dr);
	   fChain->SetBranchAddress("b1yy.dr", &b1yy_dr, &b_b1yy_dr);
	   fChain->SetBranchAddress("b2yy.dr", &b2yy_dr, &b_b2yy_dr);
	   fChain->SetBranchAddress("y1bb.dr", &y1bb_dr, &b_y1bb_dr);
	   fChain->SetBranchAddress("y2bb.dr", &y2bb_dr, &b_y2bb_dr);
	   fChain->SetBranchAddress("AddJ.pt", &AddJ_pt, &b_AddJ_pt);
	   fChain->SetBranchAddress("AddJ.eta", &AddJ_eta, &b_AddJ_eta);
	   fChain->SetBranchAddress("AddJ.phi", &AddJ_phi, &b_AddJ_phi);
	   fChain->SetBranchAddress("AddJ.m", &AddJ_m, &b_AddJ_m);
	   fChain->SetBranchAddress("AddJ.N", &AddJ_N, &b_AddJ_N);
	   fChain->SetBranchAddress("Jet.N", &Jet_N, &b_Jet_N);
	   fChain->SetBranchAddress("AddBJ.N", &AddBJ_N, &b_AddBJ_N);
   fChain->SetBranchAddress("yy.HT_l1", &yy_HT_l1, &b_yy_HT_l1);
   fChain->SetBranchAddress("yy.HT_l2", &yy_HT_l2, &b_yy_HT_l2);
   fChain->SetBranchAddress("yy.HT_l3", &yy_HT_l3, &b_yy_HT_l3);
   fChain->SetBranchAddress("yy.HT_l4", &yy_HT_l4, &b_yy_HT_l4);
   fChain->SetBranchAddress("yy.HT_l5", &yy_HT_l5, &b_yy_HT_l5);
   fChain->SetBranchAddress("yy.HP_l1", &yy_HP_l1, &b_yy_HP_l1);
   fChain->SetBranchAddress("yy.HP_l2", &yy_HP_l2, &b_yy_HP_l2);
   fChain->SetBranchAddress("yy.HP_l3", &yy_HP_l3, &b_yy_HP_l3);
   fChain->SetBranchAddress("yy.HP_l4", &yy_HP_l4, &b_yy_HP_l4);
   fChain->SetBranchAddress("yy.HP_l5", &yy_HP_l5, &b_yy_HP_l5);
   fChain->SetBranchAddress("yy.HZ_l1", &yy_HZ_l1, &b_yy_HZ_l1);
   fChain->SetBranchAddress("yy.HZ_l2", &yy_HZ_l2, &b_yy_HZ_l2);
   fChain->SetBranchAddress("yy.HZ_l3", &yy_HZ_l3, &b_yy_HZ_l3);
   fChain->SetBranchAddress("yy.HZ_l4", &yy_HZ_l4, &b_yy_HZ_l4);
   fChain->SetBranchAddress("yy.HZ_l5", &yy_HZ_l5, &b_yy_HZ_l5);
   fChain->SetBranchAddress("yy.HY_l1", &yy_HY_l1, &b_yy_HY_l1);
   fChain->SetBranchAddress("yy.HY_l2", &yy_HY_l2, &b_yy_HY_l2);
   fChain->SetBranchAddress("yy.HY_l3", &yy_HY_l3, &b_yy_HY_l3);
   fChain->SetBranchAddress("yy.HY_l4", &yy_HY_l4, &b_yy_HY_l4);
   fChain->SetBranchAddress("yy.HY_l5", &yy_HY_l5, &b_yy_HY_l5);
   fChain->SetBranchAddress("yy.HS_l1", &yy_HS_l1, &b_yy_HS_l1);
   fChain->SetBranchAddress("yy.HS_l2", &yy_HS_l2, &b_yy_HS_l2);
   fChain->SetBranchAddress("yy.HS_l3", &yy_HS_l3, &b_yy_HS_l3);
   fChain->SetBranchAddress("yy.HS_l4", &yy_HS_l4, &b_yy_HS_l4);
   fChain->SetBranchAddress("yy.HS_l5", &yy_HS_l5, &b_yy_HS_l5);
   fChain->SetBranchAddress("yy.H1_l1", &yy_H1_l1, &b_yy_H1_l1);
   fChain->SetBranchAddress("yy.H1_l2", &yy_H1_l2, &b_yy_H1_l2);
   fChain->SetBranchAddress("yy.H1_l3", &yy_H1_l3, &b_yy_H1_l3);
   fChain->SetBranchAddress("yy.H1_l4", &yy_H1_l4, &b_yy_H1_l4);
   fChain->SetBranchAddress("yy.H1_l5", &yy_H1_l5, &b_yy_H1_l5);
   fChain->SetBranchAddress("bb.HT_l1", &bb_HT_l1, &b_bb_HT_l1);
   fChain->SetBranchAddress("bb.HT_l2", &bb_HT_l2, &b_bb_HT_l2);
   fChain->SetBranchAddress("bb.HT_l3", &bb_HT_l3, &b_bb_HT_l3);
   fChain->SetBranchAddress("bb.HT_l4", &bb_HT_l4, &b_bb_HT_l4);
   fChain->SetBranchAddress("bb.HT_l5", &bb_HT_l5, &b_bb_HT_l5);
   fChain->SetBranchAddress("bb.HP_l1", &bb_HP_l1, &b_bb_HP_l1);
   fChain->SetBranchAddress("bb.HP_l2", &bb_HP_l2, &b_bb_HP_l2);
   fChain->SetBranchAddress("bb.HP_l3", &bb_HP_l3, &b_bb_HP_l3);
   fChain->SetBranchAddress("bb.HP_l4", &bb_HP_l4, &b_bb_HP_l4);
   fChain->SetBranchAddress("bb.HP_l5", &bb_HP_l5, &b_bb_HP_l5);
   fChain->SetBranchAddress("bb.HZ_l1", &bb_HZ_l1, &b_bb_HZ_l1);
   fChain->SetBranchAddress("bb.HZ_l2", &bb_HZ_l2, &b_bb_HZ_l2);
   fChain->SetBranchAddress("bb.HZ_l3", &bb_HZ_l3, &b_bb_HZ_l3);
   fChain->SetBranchAddress("bb.HZ_l4", &bb_HZ_l4, &b_bb_HZ_l4);
   fChain->SetBranchAddress("bb.HZ_l5", &bb_HZ_l5, &b_bb_HZ_l5);
   fChain->SetBranchAddress("bb.HY_l1", &bb_HY_l1, &b_bb_HY_l1);
   fChain->SetBranchAddress("bb.HY_l2", &bb_HY_l2, &b_bb_HY_l2);
   fChain->SetBranchAddress("bb.HY_l3", &bb_HY_l3, &b_bb_HY_l3);
   fChain->SetBranchAddress("bb.HY_l4", &bb_HY_l4, &b_bb_HY_l4);
   fChain->SetBranchAddress("bb.HY_l5", &bb_HY_l5, &b_bb_HY_l5);
   fChain->SetBranchAddress("bb.HS_l1", &bb_HS_l1, &b_bb_HS_l1);
   fChain->SetBranchAddress("bb.HS_l2", &bb_HS_l2, &b_bb_HS_l2);
   fChain->SetBranchAddress("bb.HS_l3", &bb_HS_l3, &b_bb_HS_l3);
   fChain->SetBranchAddress("bb.HS_l4", &bb_HS_l4, &b_bb_HS_l4);
   fChain->SetBranchAddress("bb.HS_l5", &bb_HS_l5, &b_bb_HS_l5);
   fChain->SetBranchAddress("bb.H1_l1", &bb_H1_l1, &b_bb_H1_l1);
   fChain->SetBranchAddress("bb.H1_l2", &bb_H1_l2, &b_bb_H1_l2);
   fChain->SetBranchAddress("bb.H1_l3", &bb_H1_l3, &b_bb_H1_l3);
   fChain->SetBranchAddress("bb.H1_l4", &bb_H1_l4, &b_bb_H1_l4);
   fChain->SetBranchAddress("bb.H1_l5", &bb_H1_l5, &b_bb_H1_l5);
   fChain->SetBranchAddress("yybb.HT_l1", &yybb_HT_l1, &b_yybb_HT_l1);
   fChain->SetBranchAddress("yybb.HT_l2", &yybb_HT_l2, &b_yybb_HT_l2);
   fChain->SetBranchAddress("yybb.HT_l3", &yybb_HT_l3, &b_yybb_HT_l3);
   fChain->SetBranchAddress("yybb.HT_l4", &yybb_HT_l4, &b_yybb_HT_l4);
   fChain->SetBranchAddress("yybb.HT_l5", &yybb_HT_l5, &b_yybb_HT_l5);
   fChain->SetBranchAddress("yybb.HP_l1", &yybb_HP_l1, &b_yybb_HP_l1);
   fChain->SetBranchAddress("yybb.HP_l2", &yybb_HP_l2, &b_yybb_HP_l2);
   fChain->SetBranchAddress("yybb.HP_l3", &yybb_HP_l3, &b_yybb_HP_l3);
   fChain->SetBranchAddress("yybb.HP_l4", &yybb_HP_l4, &b_yybb_HP_l4);
   fChain->SetBranchAddress("yybb.HP_l5", &yybb_HP_l5, &b_yybb_HP_l5);
   fChain->SetBranchAddress("yybb.HZ_l1", &yybb_HZ_l1, &b_yybb_HZ_l1);
   fChain->SetBranchAddress("yybb.HZ_l2", &yybb_HZ_l2, &b_yybb_HZ_l2);
   fChain->SetBranchAddress("yybb.HZ_l3", &yybb_HZ_l3, &b_yybb_HZ_l3);
   fChain->SetBranchAddress("yybb.HZ_l4", &yybb_HZ_l4, &b_yybb_HZ_l4);
   fChain->SetBranchAddress("yybb.HZ_l5", &yybb_HZ_l5, &b_yybb_HZ_l5);
   fChain->SetBranchAddress("yybb.HY_l1", &yybb_HY_l1, &b_yybb_HY_l1);
   fChain->SetBranchAddress("yybb.HY_l2", &yybb_HY_l2, &b_yybb_HY_l2);
   fChain->SetBranchAddress("yybb.HY_l3", &yybb_HY_l3, &b_yybb_HY_l3);
   fChain->SetBranchAddress("yybb.HY_l4", &yybb_HY_l4, &b_yybb_HY_l4);
   fChain->SetBranchAddress("yybb.HY_l5", &yybb_HY_l5, &b_yybb_HY_l5);
   fChain->SetBranchAddress("yybb.HS_l1", &yybb_HS_l1, &b_yybb_HS_l1);
   fChain->SetBranchAddress("yybb.HS_l2", &yybb_HS_l2, &b_yybb_HS_l2);
   fChain->SetBranchAddress("yybb.HS_l3", &yybb_HS_l3, &b_yybb_HS_l3);
   fChain->SetBranchAddress("yybb.HS_l4", &yybb_HS_l4, &b_yybb_HS_l4);
   fChain->SetBranchAddress("yybb.HS_l5", &yybb_HS_l5, &b_yybb_HS_l5);
   fChain->SetBranchAddress("yybb.H1_l1", &yybb_H1_l1, &b_yybb_H1_l1);
   fChain->SetBranchAddress("yybb.H1_l2", &yybb_H1_l2, &b_yybb_H1_l2);
   fChain->SetBranchAddress("yybb.H1_l3", &yybb_H1_l3, &b_yybb_H1_l3);
   fChain->SetBranchAddress("yybb.H1_l4", &yybb_H1_l4, &b_yybb_H1_l4);
   fChain->SetBranchAddress("yybb.H1_l5", &yybb_H1_l5, &b_yybb_H1_l5);
   fChain->SetBranchAddress("hh.HT_l1", &hh_HT_l1, &b_hh_HT_l1);
   fChain->SetBranchAddress("hh.HT_l2", &hh_HT_l2, &b_hh_HT_l2);
   fChain->SetBranchAddress("hh.HT_l3", &hh_HT_l3, &b_hh_HT_l3);
   fChain->SetBranchAddress("hh.HT_l4", &hh_HT_l4, &b_hh_HT_l4);
   fChain->SetBranchAddress("hh.HT_l5", &hh_HT_l5, &b_hh_HT_l5);
   fChain->SetBranchAddress("hh.HP_l1", &hh_HP_l1, &b_hh_HP_l1);
   fChain->SetBranchAddress("hh.HP_l2", &hh_HP_l2, &b_hh_HP_l2);
   fChain->SetBranchAddress("hh.HP_l3", &hh_HP_l3, &b_hh_HP_l3);
   fChain->SetBranchAddress("hh.HP_l4", &hh_HP_l4, &b_hh_HP_l4);
   fChain->SetBranchAddress("hh.HP_l5", &hh_HP_l5, &b_hh_HP_l5);
   fChain->SetBranchAddress("hh.HZ_l1", &hh_HZ_l1, &b_hh_HZ_l1);
   fChain->SetBranchAddress("hh.HZ_l2", &hh_HZ_l2, &b_hh_HZ_l2);
   fChain->SetBranchAddress("hh.HZ_l3", &hh_HZ_l3, &b_hh_HZ_l3);
   fChain->SetBranchAddress("hh.HZ_l4", &hh_HZ_l4, &b_hh_HZ_l4);
   fChain->SetBranchAddress("hh.HZ_l5", &hh_HZ_l5, &b_hh_HZ_l5);
   fChain->SetBranchAddress("hh.HY_l1", &hh_HY_l1, &b_hh_HY_l1);
   fChain->SetBranchAddress("hh.HY_l2", &hh_HY_l2, &b_hh_HY_l2);
   fChain->SetBranchAddress("hh.HY_l3", &hh_HY_l3, &b_hh_HY_l3);
   fChain->SetBranchAddress("hh.HY_l4", &hh_HY_l4, &b_hh_HY_l4);
   fChain->SetBranchAddress("hh.HY_l5", &hh_HY_l5, &b_hh_HY_l5);
   fChain->SetBranchAddress("hh.HS_l1", &hh_HS_l1, &b_hh_HS_l1);
   fChain->SetBranchAddress("hh.HS_l2", &hh_HS_l2, &b_hh_HS_l2);
   fChain->SetBranchAddress("hh.HS_l3", &hh_HS_l3, &b_hh_HS_l3);
   fChain->SetBranchAddress("hh.HS_l4", &hh_HS_l4, &b_hh_HS_l4);
   fChain->SetBranchAddress("hh.HS_l5", &hh_HS_l5, &b_hh_HS_l5);
   fChain->SetBranchAddress("hh.H1_l1", &hh_H1_l1, &b_hh_H1_l1);
   fChain->SetBranchAddress("hh.H1_l2", &hh_H1_l2, &b_hh_H1_l2);
   fChain->SetBranchAddress("hh.H1_l3", &hh_H1_l3, &b_hh_H1_l3);
   fChain->SetBranchAddress("hh.H1_l4", &hh_H1_l4, &b_hh_H1_l4);
   fChain->SetBranchAddress("hh.H1_l5", &hh_H1_l5, &b_hh_H1_l5);
	   fChain->SetBranchAddress("HT", &HT, &b_HT);
	   fChain->SetBranchAddress("HS", &HS, &b_HS);
	   fChain->SetBranchAddress("HP", &HP, &b_HP);
	   fChain->SetBranchAddress("H1", &H1, &b_H1);
	   fChain->SetBranchAddress("HY", &HY, &b_HY);
	   fChain->SetBranchAddress("L", &L, &b_L);
	   fChain->SetBranchAddress("met.px", &met_px, &b_met_px);
	   fChain->SetBranchAddress("met.py", &met_py, &b_met_py);
	   fChain->SetBranchAddress("met.e", &met_e, &b_met_e);
	   fChain->SetBranchAddress("met.name", &met_name, &b_met_name);
	   fChain->SetBranchAddress("Event.TotWeight", &Event_TotWeight, &b_Event_TotWeight);
	   fChain->SetBranchAddress("Event.Number", &Event_Number, &b_Event_Number);
	   fChain->SetBranchAddress("Event.bbyyWeight", &Event_bbyyWeight, &b_Event_bbyyWeight);
	   fChain->SetBranchAddress("Event.Weight", &Event_Weight, &b_Event_Weight);
	   fChain->SetBranchAddress("Event.Xsec", &Event_Xsec, &b_Event_Xsec);
	   fChain->SetBranchAddress("Event.Lumi", &Event_Lumi, &b_Event_Lumi);
	   fChain->SetBranchAddress("Event.N", &Event_N, &b_Event_N);
	   fChain->SetBranchAddress("Event.Cat", &Event_Cat, &b_Event_Cat);

		return fChain;
	
}

void Plot(TString name)
{
	histo_MYY = new TH1F("MYY","",10,120,130);
	histo_MBB = new TH1F("MBB","",10,90,140);
	histo_AddPt = new TH1F("AddPt","",100,0,100);

	histo_bb_H1_l1 = new TH1F("bb_H1_l1","",20,-1,1);
	histo_bb_HY_l3 = new TH1F("bb_HY_l3","",20,-1,1);
	histo_hh_H1_l3 = new TH1F("hh_H1_l3","",20,-1,1);
	histo_hh_HT_l1 = new TH1F("hh_HT_l1","",20,-1,1);
	histo_hh_HP_l2 = new TH1F("hh_HP_l2","",20,-1,1);
	histo_yy_HP_l1 = new TH1F("yy_HP_l1","",20,-1,1);
	histo_yy_H1_l3 = new TH1F("yy_H1_l3","",20,-1,1);

	for(int i = 0; i<19; i++)
	{
		histo_HT[i] = new TH1F(Form("histo_HT_l%i",i+1),"",100,-1,1);
		histo_HP[i] = new TH1F(Form("histo_HP_l%i",i+1),"",100,-1,1);
		histo_HS[i] = new TH1F(Form("histo_HS_l%i",i+1),"",100,-2,40);
		histo_H1[i] = new TH1F(Form("histo_H1_l%i",i+1),"",100,-1,1);
		histo_HY[i] = new TH1F(Form("histo_HY_l%i",i+1),"",100,-1,1);

		histos.push_back(histo_HT[i]);
		histos.push_back(histo_HP[i]);
		histos.push_back(histo_HS[i]);
		histos.push_back(histo_H1[i]);
		histos.push_back(histo_HY[i]);
	}
	
	histo_L[0] = new TH1F("histo_L_l0_l1","",100,0,2);
	histo_L[1] = new TH1F("histo_L_l1_l2","",100,0,2);
	histo_L[2] = new TH1F("histo_L_l2_l3","",100,0,2);
	histo_L[3] = new TH1F("histo_L_l3_l4","",100,0,2);
	histo_L[4] = new TH1F("histo_L_l4_l5","",100,0,2);
	histo_L[5] = new TH1F("histo_L_l2_l0","",100,0,2);

	histos.push_back(histo_L[0]);
	histos.push_back(histo_L[1]);
	histos.push_back(histo_L[2]);
	histos.push_back(histo_L[3]);
	histos.push_back(histo_L[4]);
	histos.push_back(histo_L[5]);
	

	double sumW = 0;
	
	double sumW1 = 0;
	double sumW2 = 0;
	double sumW3 = 0;
	double sumW4 = 0;

	TChain* chain = GetTChain(name);
	Int_t nentries = chain->GetEntriesFast();
	for(int i = 0; i<nentries; i++)
	{
		int nb = chain->GetEntry(i);
		
		//if (Event_Cat != 4) continue;

		if(!(yy_m*0.001 >= 120 && yy_m*0.001 <= 130)){
			histo_bb_H1_l1->Fill(bb_H1_l1,Event_TotWeight);
			histo_bb_HY_l3->Fill(bb_HY_l3,Event_TotWeight);
			histo_hh_H1_l3->Fill(hh_H1_l3,Event_TotWeight);
			histo_hh_HT_l1->Fill(hh_HT_l1,Event_TotWeight);
			histo_hh_HP_l2->Fill(hh_HP_l2,Event_TotWeight);
			histo_yy_HP_l1->Fill(yy_HP_l1,Event_TotWeight);
			histo_yy_H1_l3->Fill(yy_H1_l3,Event_TotWeight);
		}
		for(int l = 0; l<HT->size(); l++)
		{
			histo_HT[l]->Fill(HT->at(l));
			histo_HP[l]->Fill(HP->at(l));
			histo_HS[l]->Fill(HS->at(l));
			histo_H1[l]->Fill(H1->at(l));
			histo_HY[l]->Fill(HY->at(l));
		} 

		histo_L[0]->Fill(fabs(L->at(0)));
		histo_L[1]->Fill(fabs(L->at(1)));
		histo_L[2]->Fill(fabs(L->at(2)));
		histo_L[3]->Fill(fabs(L->at(3)));
		histo_L[4]->Fill(fabs(L->at(4)));
		histo_L[5]->Fill(fabs((1./L->at(0)) * (1./L->at(1))));

		histo_AddPt->Fill(AddJ_pt*1.e-6);

		histo_MYY->Fill(yy_m*0.001,Event_TotWeight);
		histo_MBB->Fill(bb_m*0.001);

		if(yy_m*0.001 >= 105 && yy_m*0.001 <= 160){
		if (Event_Cat == 1) sumW1 += Event_TotWeight;
		if (Event_Cat == 2) sumW2 += Event_TotWeight;
		if (Event_Cat == 3) sumW3 += Event_TotWeight;
		if (Event_Cat == 4) sumW4 += Event_TotWeight;
		
		sumW += Event_TotWeight;
		}
	}
		
	for(int i = 0; i<histos.size(); i++)
	{
		histos.at(i)->Scale(1./histos.at(i)->Integral());
	}
		cout<<"Total Weight : "<<sumW<<endl;
		cout<<"Weight Cat 1: "<<sumW1<<endl;
		cout<<"Weight Cat 2: "<<sumW2<<endl;
		cout<<"Weight Cat 3: "<<sumW3<<endl;
		cout<<"Weight Cat 4: "<<sumW4<<endl;

		cout<<"Integral : "<<histo_MYY->Integral()<<endl;
		histo_AddPt->Scale(1./histo_AddPt->Integral());
}

int main(int argc, char* argv [])
{
	TString name = argv[1];
	TString outputname = argv[2];
	Plot(name);
	TFile* _output = new TFile(outputname,"RECREATE");
	for(int i = 0; i<histos.size(); i++)
	{
		histos.at(i)->SetDirectory(_output);
	}

	histo_bb_H1_l1->Scale(1./histo_bb_H1_l1->Integral());
	histo_bb_HY_l3->Scale(1./histo_bb_HY_l3->Integral());
	histo_hh_H1_l3->Scale(1./histo_hh_H1_l3->Integral());
	histo_hh_HT_l1->Scale(1./histo_hh_HT_l1->Integral());
	histo_hh_HP_l2->Scale(1./histo_hh_HP_l2->Integral());
	histo_yy_HP_l1->Scale(1./histo_yy_HP_l1->Integral());
	histo_yy_H1_l3->Scale(1./histo_yy_H1_l3->Integral());

	histo_AddPt->SetDirectory(_output);
	histo_MYY->SetDirectory(_output);
	histo_MBB->SetDirectory(_output);
	histo_bb_H1_l1->SetDirectory(_output);
	histo_bb_HY_l3->SetDirectory(_output);
	histo_hh_H1_l3->SetDirectory(_output);
	histo_hh_HT_l1->SetDirectory(_output);
	histo_hh_HP_l2->SetDirectory(_output);
	histo_yy_HP_l1->SetDirectory(_output);
	histo_yy_H1_l3->SetDirectory(_output);

	_output->Write();
	_output->Close();
	return 1.;

}

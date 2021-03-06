#include <memory>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>

// ROOT include(s):
#include <TFile.h>
#include <TH1F.h>
#include <TChain.h>
#include <TError.h>
#include <TString.h>
#include <TStopwatch.h>
#include <TSystem.h>
#include "TLorentzVector.h"
#include "TMath.h"
#include "TROOT.h"
#include "TVector3.h"

// Infrastructure include(s):
#ifdef ROOTCORE
#   include "xAODRootAccess/Init.h"
#   include "xAODRootAccess/TEvent.h"
#endif // ROOTCORE

// EDM include(s):
#include "xAODEventInfo/EventInfo.h"

// Local include(s):
#include "CPAnalysisExamples/errorcheck.h"

using namespace std;

TChain* GetChains(std::vector<TString> samples, TString TreeName);
TChain* Load();
int GetNTuple(int i, int n,TString out);
void SetBranches(TChain* chain);
void SetLumi(TChain* fchain);
bool Ana();
void NewEvents();
void ClearQuantities();
void ComputeQuantities();
void FillFoxMomenta(vector<TLorentzVector> Bjets, int N);
double FoxMomenta_l(vector<TLorentzVector> Bjets, TString type, int l);
double getCosOmga(int i, int j, vector<TLorentzVector> Bjets);
double getWight(int i, int j, vector<TLorentzVector> Bjets, TString type);
double getLegendreFactor(int l, double x);
void SetOuput(TString out, int n);
void LinkOutputBranches(TTree *_output_tree);
void CalibrateBJet();

vector<TVector3> GenerateT();
double sumTdotP(TVector3 T, vector<TLorentzVector> P);
TVector3 GetThrust(vector<TLorentzVector> P);
void FillLMomenta(vector<TLorentzVector> Bjets, int N, TVector3 T);
double LMomenta_l(vector<TLorentzVector> Bjets, int l, TVector3 T);

double getNEventPerFile(TFile* file, TString histoname);

void getNEventPerDirectory(std::vector<TString> Samples, TString histo, double &n_mc16a, double &n_mc16d, double &n_mc16e);

void getNEventPerDirectory(std::vector<TString> Samples, TString histo, double &n_mc16a, double &n_mc16d, double &n_mc16e)
{
	for(int i = 0; i<Samples.size(); i++)
	{
		TFile* file_i = new TFile(Samples[i]);	
		double n = getNEventPerFile(file_i, histo);
		if(Samples[i].Contains("mc16a"))
		{
			//std::cout<<n<<std::endl;
			n_mc16a += n;
		}
		if(Samples[i].Contains("mc16d"))
		{
			n_mc16d += n;
		}
		if(Samples[i].Contains("mc16e"))
		{
			n_mc16e += n;
		}
	}
}

double getNEventPerFile(TFile* file, TString name){
	
	TH1F *histo = (TH1F*) file->Get(name);

	double bin1 = histo->GetBinContent(1);
	double bin2 = histo->GetBinContent(2);
	double bin3 = histo->GetBinContent(3);

	return bin3*(bin1/bin2);

}
const char* APP_NAME;

TFile *_output_file; 
TTree *_output1_tree; 
TTree *_output2_tree;
TTree *_output3_tree;
TTree *_output4_tree;

vector<float> *_jets_pt;
vector<float> *_jets_eta;
vector<float> *_jets_phi;
vector<float> *_jets_m;

vector<float> *_OneMu_pt;
vector<float> *_OneMu_eta;
vector<float> *_OneMu_phi;
vector<float> *_OneMu_m;
vector<float> *_PtReco_SF;
vector<float> *_Reg_SF;

vector<int> *_jets_score;
vector<char> *_jets_btag_60;
vector<char> *_jets_btag_77;
vector<char> *_jets_btag_70;
vector<char> *_jets_btag_85;

vector<float> *_gamma_pt;
vector<float> *_gamma_eta;
vector<float> *_gamma_phi;
vector<float> *_gamma_m;

vector<double> *_met_px;
vector<double> *_met_py;
vector<double> *_met_e;
vector<string> *_met_name;

Char_t    _event_ispass;
ULong64_t    _event_number;
Float_t   _event_bbyy_weight;
Float_t   _event_weight;
Float_t   _event_xsec;
Int_t     _event_njets;
Float_t   _event_lumi;
Float_t   _nevent;
Float_t   _event_total_weight;
Int_t     _event_cat;
Int_t     _BDT_cat;

Int_t     _event_N_lep;
Int_t     _event_N_j_central;
Int_t     _event_N_j_btag;

bool debug = false;

vector<TLorentzVector>         _photons;
vector<TLorentzVector>         _jets;
TLorentzVector                 _Hyy;
TLorentzVector                 _Hbb;
TLorentzVector                 _HH;
TLorentzVector                 _SumJ;
vector<TLorentzVector>         _Bjets; 
vector<TVector3>               _Thrust;

double _n_mc16a;
double _n_mc16d;
double _n_mc16e;

double  _pt_b1;
double _eta_b1;
double _phi_b1;
double   _e_b1;

double  _pt_b2;
double _eta_b2;
double _phi_b2;
double   _e_b2;

double _sf_b1;
double _sf_b2;

int _btag_ds;

double  _pt_y1;
double _eta_y1;
double _phi_y1;
double   _e_y1;

double  _pt_y2;
double _eta_y2;
double _phi_y2;
double   _e_y2;

double   _m_hh;
double  _pt_hh;
double _eta_hh;
double _phi_hh;
double   _e_hh;

double   _m_bb;
double  _pt_bb;
double _eta_bb;
double _phi_bb;
double   _e_bb;

double   _m_yy;
double  _pt_yy;
double _eta_yy;
double _phi_yy;
double   _e_yy;

double  _dr_yy;
double  _dr_bb;
double  _dr_hh;
double  _dr_y1bb;
double  _dr_y2bb;
double  _dr_b1yy;
double  _dr_b2yy;

double _pt_jets;
double _eta_jets;
double _phi_jets;
double _m_jets;

int    _n_jets;
int    _n_add_j;
int    _n_add_bjet;

double _score_b1;
double _score_b2;

double   _met_TST;
double   _met_Soft;

double _Y_HT_l1;
double _Y_HT_l2;
double _Y_HT_l3;
double _Y_HT_l4;
double _Y_HT_l5;

double _Y_HP_l1;
double _Y_HP_l2;
double _Y_HP_l3;
double _Y_HP_l4;
double _Y_HP_l5;

double _Y_HY_l1;
double _Y_HY_l2;
double _Y_HY_l3;
double _Y_HY_l4;
double _Y_HY_l5;

double _Y_HZ_l1;
double _Y_HZ_l2;
double _Y_HZ_l3;
double _Y_HZ_l4;
double _Y_HZ_l5;

double _Y_HS_l1;
double _Y_HS_l2;
double _Y_HS_l3;
double _Y_HS_l4;
double _Y_HS_l5;

double _Y_H1_l1;
double _Y_H1_l2;
double _Y_H1_l3;
double _Y_H1_l4;
double _Y_H1_l5;

double _B_HT_l1;
double _B_HT_l2;
double _B_HT_l3;
double _B_HT_l4;
double _B_HT_l5;

double _B_HP_l1;
double _B_HP_l2;
double _B_HP_l3;
double _B_HP_l4;
double _B_HP_l5;

double _B_HY_l1;
double _B_HY_l2;
double _B_HY_l3;
double _B_HY_l4;
double _B_HY_l5;

double _B_HZ_l1;
double _B_HZ_l2;
double _B_HZ_l3;
double _B_HZ_l4;
double _B_HZ_l5;

double _B_HS_l1;
double _B_HS_l2;
double _B_HS_l3;
double _B_HS_l4;
double _B_HS_l5;

double _B_H1_l1;
double _B_H1_l2;
double _B_H1_l3;
double _B_H1_l4;
double _B_H1_l5;

double _YB_HT_l1;
double _YB_HT_l2;
double _YB_HT_l3;
double _YB_HT_l4;
double _YB_HT_l5;

double _YB_HP_l1;
double _YB_HP_l2;
double _YB_HP_l3;
double _YB_HP_l4;
double _YB_HP_l5;

double _YB_HY_l1;
double _YB_HY_l2;
double _YB_HY_l3;
double _YB_HY_l4;
double _YB_HY_l5;

double _YB_HZ_l1;
double _YB_HZ_l2;
double _YB_HZ_l3;
double _YB_HZ_l4;
double _YB_HZ_l5;

double _YB_HS_l1;
double _YB_HS_l2;
double _YB_HS_l3;
double _YB_HS_l4;
double _YB_HS_l5;

double _YB_H1_l1;
double _YB_H1_l2;
double _YB_H1_l3;
double _YB_H1_l4;
double _YB_H1_l5;

double _HH_HT_l1;
double _HH_HT_l2;
double _HH_HT_l3;
double _HH_HT_l4;
double _HH_HT_l5;

double _HH_HP_l1;
double _HH_HP_l2;
double _HH_HP_l3;
double _HH_HP_l4;
double _HH_HP_l5;

double _HH_HY_l1;
double _HH_HY_l2;
double _HH_HY_l3;
double _HH_HY_l4;
double _HH_HY_l5;

double _HH_HZ_l1;
double _HH_HZ_l2;
double _HH_HZ_l3;
double _HH_HZ_l4;
double _HH_HZ_l5;

double _HH_HS_l1;
double _HH_HS_l2;
double _HH_HS_l3;
double _HH_HS_l4;
double _HH_HS_l5;

double _HH_H1_l1;
double _HH_H1_l2;
double _HH_H1_l3;
double _HH_H1_l4;
double _HH_H1_l5;

double _add_pt_test;

vector<double> _H_S;
vector<double> _H_T;
vector<double> _H_P;
vector<double> _H_Z;
vector<double> _H_1;
vector<double> _H_Y;

vector<double> _L;

vector<int>    _jet_rank;

void ClearQuantities()
{

   _add_pt_test = 0.0;

   _sf_b1  = 0.0;
   _sf_b2  = 0.0;

   _met_TST   = 0.0;
   _met_Soft  = 0.0;

   _btag_ds= 0;
   _pt_b1  = 0.0;
   _eta_b1 = 0.0;
   _phi_b1 = 0.0;
   _e_b1   = 0.0;

   _pt_b2  = 0.0;
   _eta_b2 = 0.0;
   _phi_b2 = 0.0;
   _e_b2   = 0.0;

   _pt_y1  = 0.0;
   _eta_y1 = 0.0;
   _phi_y1 = 0.0;
   _e_y1   = 0.0;

   _pt_y2  = 0.0;
   _eta_y2 = 0.0;
   _phi_y2 = 0.0;
   _e_y2   = 0.0;

   _m_hh   = 0.0;
   _pt_hh  = 0.0;
   _eta_hh = 0.0;
   _phi_hh = 0.0;
   _e_hh   = 0.0;

   _m_bb   = 0.0;
   _pt_bb  = 0.0;
   _eta_bb = 0.0;
   _phi_bb = 0.0;
   _e_bb   = 0.0;

   _m_yy   = 0.0;
   _pt_yy  = 0.0;
   _eta_yy = 0.0;
   _phi_yy = 0.0;
   _e_yy   = 0.0;

  _dr_yy   = 0.0;
  _dr_bb   = 0.0;
  _dr_hh   = 0.0;
  _dr_y1bb = 0.0;
  _dr_y2bb = 0.0;
  _dr_b1yy = 0.0;
  _dr_b2yy = 0.0;

  _pt_jets  = 0.0;
  _eta_jets = 0.0;
  _phi_jets = 0.0;
  _m_jets   = 0.0;

  _n_jets   = 0.0;

  _n_add_j  = 0.0;

  _n_add_bjet = 0.0;

   _score_b1 = 0.0;
   _score_b2 = 0.0;

   _event_number = 0.0;


 _Y_HT_l1 = 0.0;
 _Y_HT_l2 = 0.0;
 _Y_HT_l3 = 0.0;
 _Y_HT_l4 = 0.0;
 _Y_HT_l5 = 0.0;

 _Y_HP_l1 = 0.0;
 _Y_HP_l2 = 0.0;
 _Y_HP_l3 = 0.0;
 _Y_HP_l4 = 0.0;
 _Y_HP_l5 = 0.0;

 _Y_HY_l1 = 0.0;
 _Y_HY_l2 = 0.0;
 _Y_HY_l3 = 0.0;
 _Y_HY_l4 = 0.0;
 _Y_HY_l5 = 0.0;

 _Y_HZ_l1 = 0.0;
 _Y_HZ_l2 = 0.0;
 _Y_HZ_l3 = 0.0;
 _Y_HZ_l4 = 0.0;
 _Y_HZ_l5 = 0.0;

 _Y_HS_l1 = 0.0;
 _Y_HS_l2 = 0.0;
 _Y_HS_l3 = 0.0;
 _Y_HS_l4 = 0.0;
 _Y_HS_l5 = 0.0;

 _Y_H1_l1 = 0.0;
 _Y_H1_l2 = 0.0;
 _Y_H1_l3 = 0.0;
 _Y_H1_l4 = 0.0;
 _Y_H1_l5 = 0.0;

 _B_HT_l1 = 0.0;
 _B_HT_l2 = 0.0;
 _B_HT_l3 = 0.0;
 _B_HT_l4 = 0.0;
 _B_HT_l5 = 0.0;

 _B_HP_l1 = 0.0;
 _B_HP_l2 = 0.0;
 _B_HP_l3 = 0.0;
 _B_HP_l4 = 0.0;
 _B_HP_l5 = 0.0;

 _B_HY_l1 = 0.0;
 _B_HY_l2 = 0.0;
 _B_HY_l3 = 0.0;
 _B_HY_l4 = 0.0;
 _B_HY_l5 = 0.0;

 _B_HZ_l1 = 0.0;
 _B_HZ_l2 = 0.0;
 _B_HZ_l3 = 0.0;
 _B_HZ_l4 = 0.0;
 _B_HZ_l5 = 0.0;

 _B_HS_l1 = 0.0;
 _B_HS_l2 = 0.0;
 _B_HS_l3 = 0.0;
 _B_HS_l4 = 0.0;
 _B_HS_l5 = 0.0;

 _B_H1_l1 = 0.0;
 _B_H1_l2 = 0.0;
 _B_H1_l3 = 0.0;
 _B_H1_l4 = 0.0;
 _B_H1_l5 = 0.0;

 _YB_HT_l1 = 0.0;
 _YB_HT_l2 = 0.0;
 _YB_HT_l3 = 0.0;
 _YB_HT_l4 = 0.0;
 _YB_HT_l5 = 0.0;

 _YB_HP_l1 = 0.0;
 _YB_HP_l2 = 0.0;
 _YB_HP_l3 = 0.0;
 _YB_HP_l4 = 0.0;
 _YB_HP_l5 = 0.0;

 _YB_HY_l1 = 0.0;
 _YB_HY_l2 = 0.0;
 _YB_HY_l3 = 0.0;
 _YB_HY_l4 = 0.0;
 _YB_HY_l5 = 0.0;

 _YB_HZ_l1 = 0.0;
 _YB_HZ_l2 = 0.0;
 _YB_HZ_l3 = 0.0;
 _YB_HZ_l4 = 0.0;
 _YB_HZ_l5 = 0.0;

 _YB_HS_l1 = 0.0;
 _YB_HS_l2 = 0.0;
 _YB_HS_l3 = 0.0;
 _YB_HS_l4 = 0.0;
 _YB_HS_l5 = 0.0;

 _YB_H1_l1 = 0.0;
 _YB_H1_l2 = 0.0;
 _YB_H1_l3 = 0.0;
 _YB_H1_l4 = 0.0;
 _YB_H1_l5 = 0.0;

 _HH_HT_l1 = 0.0;
 _HH_HT_l2 = 0.0;
 _HH_HT_l3 = 0.0;
 _HH_HT_l4 = 0.0;
 _HH_HT_l5 = 0.0;

 _HH_HP_l1 = 0.0;
 _HH_HP_l2 = 0.0;
 _HH_HP_l3 = 0.0;
 _HH_HP_l4 = 0.0;
 _HH_HP_l5 = 0.0;

 _HH_HY_l1 = 0.0;
 _HH_HY_l2 = 0.0;
 _HH_HY_l3 = 0.0;
 _HH_HY_l4 = 0.0;
 _HH_HY_l5 = 0.0;

 _HH_HZ_l1 = 0.0;
 _HH_HZ_l2 = 0.0;
 _HH_HZ_l3 = 0.0;
 _HH_HZ_l4 = 0.0;
 _HH_HZ_l5 = 0.0;

 _HH_HS_l1 = 0.0;
 _HH_HS_l2 = 0.0;
 _HH_HS_l3 = 0.0;
 _HH_HS_l4 = 0.0;
 _HH_HS_l5 = 0.0;

 _HH_H1_l1 = 0.0;
 _HH_H1_l2 = 0.0;
 _HH_H1_l3 = 0.0;
 _HH_H1_l4 = 0.0;
 _HH_H1_l5 = 0.0;

    _H_S.clear();
    _H_T.clear();
    _H_P.clear();
    _H_Z.clear();
    _H_1.clear();
    _H_Y.clear();
    _L.clear();

    _jet_rank.clear();

}

vector<TVector3> GenerateT()
{
	std::vector<TVector3> rslt;
	TVector3 t;
	
	for(float theta = 0.01; theta <= 3.14; theta += 0.01)
	{
	for(float phi = -3.14; phi<= 3.14; phi +=0.01)
	{
		double eta = -log(tan(theta/2));
		t.SetPtEtaPhi(1.,eta,phi);
		rslt.push_back(t);
	}
	}
	std::cout<< rslt.size() <<" Thrust vectors generated... "<<std::endl;	
		
	return rslt;
}

double sumTdotP(TVector3 T, vector<TLorentzVector> P)
{
	double sum = 0.;

	for(int i = 0; i<P.size(); i++)
	{
		sum += T.Dot(P.at(i).Vect());
	}
	return sum;
}
TVector3 GetThrust(vector<TLorentzVector> P)
{
	double maxsum = 0.;
	int    index  = 0;
	for(int i = 0; i<_Thrust.size(); i++)
	{
		double sum = sumTdotP(_Thrust[i], P);
		if( sum < maxsum ) continue;
		maxsum = sum;
		index = i;
	}
	return _Thrust[index];
}

void FillLMomenta(vector<TLorentzVector> Bjets, int N, TVector3 T)
{
	for(int l = 0; l<N-1; l++)
	{
		
		_L.push_back(LMomenta_l(Bjets,l,T)/LMomenta_l(Bjets,l+1,T));
	}
}

double LMomenta_l(vector<TLorentzVector> Bjets, int l, TVector3 T)
{

	double L_l = 0;
	
	for(int i = 0; i<Bjets.size(); i++)
	{
		
		L_l += Bjets.at(i).P()*getLegendreFactor(l,T.Angle(Bjets.at(i).Vect()));
	}

	return L_l;
}

void FillFoxMomenta(vector<TLorentzVector> Bjets, int N)
{
	for(int l = 1; l<N; l++)
	{
		
		_H_S.push_back(FoxMomenta_l(Bjets,'S',l));
		_H_T.push_back(FoxMomenta_l(Bjets,'T',l));
		_H_P.push_back(FoxMomenta_l(Bjets,'P',l));
		_H_Z.push_back(FoxMomenta_l(Bjets,'Z',l));
		_H_1.push_back(FoxMomenta_l(Bjets,'1',l));
		_H_Y.push_back(FoxMomenta_l(Bjets,'Y',l));
	}
}

double FoxMomenta_l(vector<TLorentzVector> Bjets, TString type,int l)
{
	double w =0.;
	if(Bjets.size() < 2)
	{
		return 0;
	}
	for(int i = 0; i<Bjets.size()-1; i++)
	{
		for(int j = i+1; j<Bjets.size(); j++)
		{
			
			w += getWight(i, j,Bjets,type)*getLegendreFactor(l,getCosOmga(i,j,Bjets));
		}
	}
	
	return w;
}

double getCosOmga(int i, int j, vector<TLorentzVector> Bjets)
{
	double Comga_ij = cos(Bjets.at(i).Theta())*cos(Bjets.at(j).Theta()) + sin(Bjets.at(i).Theta())*sin(Bjets.at(j).Theta())*cos(Bjets.at(i).Phi() - Bjets.at(j).Phi());
	/*
	if(Comga_ij > 1.)
	{
		Comga_ij = 1.;
	}
	if(Comga_ij < -1.)
	{
		Comga_ij = -1.;
	}
	*/
	return Comga_ij;
}
double getWight(int i, int j, vector<TLorentzVector> Bjets, TString type)
{


	double sum_P = 0;
	double sum_T = 0;
	double sum_S = 0;
	double sum_Z = 0;
	double sum_Y = 0;
	double average = 0.5*(Bjets.at(i).Eta() + Bjets.at(j).Eta());
	for( int k = 0; k<Bjets.size(); k++)
	{
		sum_P += Bjets.at(k).P();
		sum_T += Bjets.at(k).Pt();
		sum_Z += Bjets.at(k).Pz();
		sum_S += Bjets.at(k).E();
		sum_Y += 1./fabs(Bjets.at(k).Eta() - average);
	}
	
	double w_ij;
	if(type == 'S')
	{
		w_ij = (Bjets.at(i).P()*Bjets.at(j).P())/TMath::Power(sum_S,2);
	} else if(type == 'P') 
	{
		w_ij = (Bjets.at(i).P()*Bjets.at(j).P())/TMath::Power(sum_P,2);
	} else if(type == 'T')
	{
		w_ij = (Bjets.at(i).Pt()*Bjets.at(j).Pt())/TMath::Power(sum_T,2);
	} else if(type == 'Z')
	{
		w_ij = (Bjets.at(i).Pz()*Bjets.at(j).Pz())/TMath::Power(sum_Z,2);
	} else if( type == '1')
	{
		w_ij = 1.;
	} else if( type == 'Y')
	{
				
		w_ij = (1./fabs(Bjets.at(i).Eta() - average)) * (1./fabs(Bjets.at(j).Eta() - average)) / TMath::Power(sum_Y,2);
		
	} else {
		w_ij = 0.;
	}
	return w_ij;

}
double getLegendreFactor(int n, double x)
{
	double r = 0;
	for(int k = 0; k<=n; k++)
	{
		
		double p1 = TMath::Factorial((double)n)/(TMath::Factorial((double)(n-k))*TMath::Factorial((double)k));
		double p2 = TMath::Power(x-1,(double)(n-k))*TMath::Power(x+1,(double)k);
		r += p1*p1*p2;
	}
	
	r /= TMath::Power(2.,(double)n);
	return r;
}

void NewEvents()
{
_photons.clear();
_jets.clear();
_Bjets.clear();
_Hyy(0);
_Hbb(0);
_SumJ.SetPtEtaPhiE(0,0,0,0);
_HH(0);
}

void SetBranches(TChain* fchain)
{

	fchain->SetBranchAddress("HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.pt",                    &_jets_pt);
	fchain->SetBranchAddress("HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.eta",                   &_jets_eta);
	fchain->SetBranchAddress("HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.phi",                   &_jets_phi);
	fchain->SetBranchAddress("HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.m",                     &_jets_m);

	fchain->SetBranchAddress("HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.OneMu_pt",                    &_OneMu_pt);
	fchain->SetBranchAddress("HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.OneMu_eta",                   &_OneMu_eta);
	fchain->SetBranchAddress("HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.OneMu_phi",                   &_OneMu_phi);
	fchain->SetBranchAddress("HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.OneMu_m",                     &_OneMu_m);
	fchain->SetBranchAddress("HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.PtRecoSF",                    &_PtReco_SF);
	fchain->SetBranchAddress("HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.NNRegPtSF",                       &_Reg_SF);     

	fchain->SetBranchAddress("HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.DL1r_bin",   &_jets_score);
	fchain->SetBranchAddress("HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.DL1r_FixedCutBEff_60",&_jets_btag_60);
	fchain->SetBranchAddress("HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.DL1r_FixedCutBEff_70",&_jets_btag_70);
	fchain->SetBranchAddress("HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.DL1r_FixedCutBEff_77",&_jets_btag_77);
	fchain->SetBranchAddress("HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.DL1r_FixedCutBEff_85",&_jets_btag_85);

	fchain->SetBranchAddress("HGamPhotonsAuxDyn.pt", &_gamma_pt);
	fchain->SetBranchAddress("HGamPhotonsAuxDyn.eta",&_gamma_eta);
	fchain->SetBranchAddress("HGamPhotonsAuxDyn.phi",&_gamma_phi);
	fchain->SetBranchAddress("HGamPhotonsAuxDyn.m",  &_gamma_m);

	fchain->SetBranchAddress("HGamMET_Reference_AntiKt4EMPFlowAuxDyn.mpx",  &_met_px);
	fchain->SetBranchAddress("HGamMET_Reference_AntiKt4EMPFlowAuxDyn.mpy",  &_met_py);
	fchain->SetBranchAddress("HGamMET_Reference_AntiKt4EMPFlowAuxDyn.sumet",&_met_e);
	fchain->SetBranchAddress("HGamMET_Reference_AntiKt4EMPFlowAuxDyn.name", &_met_name);

	fchain->SetBranchAddress("HGamEventInfoAuxDyn.isPassed",                   &_event_ispass);
	fchain->SetBranchAddress("HGamEventInfoAuxDyn.yybb_weight", &_event_bbyy_weight);
	fchain->SetBranchAddress("HGamEventInfoAuxDyn.weight",                     &_event_weight);
	if(!_isData)fchain->SetBranchAddress("HGamEventInfoAuxDyn.crossSectionBRfilterEff",    &_event_xsec);
	fchain->SetBranchAddress("HGamEventInfoAuxDyn.N_j",                        &_event_njets);
	fchain->SetBranchAddress("HGamEventInfoAuxDyn.yybb_nonRes_cutBased_btag77_85_Cat",&_event_cat);
	fchain->SetBranchAddress("HGamEventInfoAuxDyn.yybb_nonRes_XGBoost_btag77_85_Cat",&_BDT_cat);
	fchain->SetBranchAddress("HGamEventInfoAuxDyn.N_lep", &_event_N_lep);
	fchain->SetBranchAddress("HGamEventInfoAuxDyn.N_j_central", &_event_N_j_central);
	fchain->SetBranchAddress("HGamEventInfoAuxDyn.N_j_btag", &_event_N_j_btag);
	

}

void SetLumi(TChain* fchain)
{
	TString name(fchain->GetFile()->GetName());
	if(!_isData)_nevent = getNEventPerFile(fchain->GetFile(), _histo_name);
	if(name.Contains("/mc16a/"))
	{
		_event_lumi = 36214.96;
		if(_isDirectory)
		{
			_nevent=_n_mc16a;	
		}
	}else if(name.Contains("/mc16d/"))
	{
		_event_lumi = 44307.4;
		if(_isDirectory)
		{
			_nevent=_n_mc16d;	
		}
	}else if(name.Contains("/mc16e/"))
	{
		_event_lumi = 58450.1;
		if(_isDirectory)
		{
			_nevent=_n_mc16e;	
		}
	}else {
		_event_lumi = 1.;
	}

	if(name.Contains("lambda06") or name.Contains("lambda00") or name.Contains("MGPy8_hh_yybb_plus_lambda10_AF2"))
	{
		if(name.Contains("/mc16d/"))
		{
		_event_lumi = 36214.96 + 44307.4;
		if(_isDirectory)
		{
			_nevent=_n_mc16d;	
		}
		}
	}
	
	if(!_isData) _event_total_weight = (_event_weight*_event_bbyy_weight*_event_xsec*_event_lumi)/_nevent;
	if(_isData)
	{
		_event_total_weight = 1.;
	}

	//std::cout<<_nevent<<std::endl;
}

void SetOuput(TString out, int n)
{
	TString output_name = "output/output";	
	output_name += _Samplename[0];
	output_name += _Type;
	output_name += _Version;
	output_name += ".root";
	TString output_name2 = "output/output";
	if(out == "splitted")
	{
		output_name2 += _Samplename[0];
		output_name2 += _Type;
		output_name2 += _Version;
		output_name2 += "_";
		output_name2 += std::to_string(n);
		output_name2 += ".root";
		output_name = output_name2;
	}
	_output_file = new TFile (output_name,"RECREATE");
	_output1_tree = new TTree("Tree_Train", "Output for DNN");
	_output2_tree = new TTree("Tree_Val", "Output for DNN");
	_output3_tree = new TTree("Tree_Test", "Output for DNN");
	_output4_tree = new TTree("Tree", "Output for DNN");
	
	LinkOutputBranches(_output1_tree);
	LinkOutputBranches(_output2_tree);
	LinkOutputBranches(_output3_tree);
	LinkOutputBranches(_output4_tree);
}

void LinkOutputBranches(TTree *_output_tree)
{
	_output_tree->Branch("b1.pt", &_pt_b1);
	_output_tree->Branch("b2.pt", &_pt_b2);
	_output_tree->Branch("b1.eta",&_eta_b1);
	_output_tree->Branch("b2.eta",&_eta_b2);
	_output_tree->Branch("b1.phi",&_phi_b1);
	_output_tree->Branch("b2.phi",&_phi_b2);
	_output_tree->Branch("b1.e",  &_e_y1);
	_output_tree->Branch("b2.e",  &_e_y2);
	_output_tree->Branch("b1.score", &_score_b1);
	_output_tree->Branch("b2.score", &_score_b2);

	_output_tree->Branch("b1.PtReco_SF", &_sf_b1);
	_output_tree->Branch("b2.PtReco_SF", &_sf_b2);

	_output_tree->Branch("y1.pt", &_pt_y1);
	_output_tree->Branch("y2.pt", &_pt_y2);
	_output_tree->Branch("y1.eta",&_eta_y1);
	_output_tree->Branch("y2.eta",&_eta_y2);
	_output_tree->Branch("y1.phi",&_phi_y1);
	_output_tree->Branch("y2.phi",&_phi_y2);
	_output_tree->Branch("y1.e",  &_e_y1);
	_output_tree->Branch("y2.e",  &_e_y2);

	_output_tree->Branch("hh.pt", &_pt_hh);
	_output_tree->Branch("yy.pt", &_pt_yy);
	_output_tree->Branch("bb.pt", &_pt_bb);
	_output_tree->Branch("hh.eta",&_eta_hh);
	_output_tree->Branch("yy.eta",&_eta_yy);
	_output_tree->Branch("bb.eta",&_eta_bb);
	_output_tree->Branch("hh.phi",&_phi_hh);
	_output_tree->Branch("yy.phi",&_phi_yy);
	_output_tree->Branch("bb.phi",&_phi_bb);
	_output_tree->Branch("hh.e",  &_e_hh);
	_output_tree->Branch("yy.e",  &_e_yy);
	_output_tree->Branch("bb.e",  &_e_bb);
	_output_tree->Branch("hh.m",  &_m_hh);
	_output_tree->Branch("yy.m",  &_m_yy);
	_output_tree->Branch("bb.m",  &_m_bb);

	_output_tree->Branch("hh.dr",  &_dr_hh);
	_output_tree->Branch("yy.dr",  &_dr_yy);
	_output_tree->Branch("bb.dr",  &_dr_bb);
	_output_tree->Branch("b1yy.dr",  &_dr_b1yy);
	_output_tree->Branch("b2yy.dr",  &_dr_b2yy);
	_output_tree->Branch("y1bb.dr",  &_dr_y1bb);
	_output_tree->Branch("y2bb.dr",  &_dr_y2bb);

	_output_tree->Branch("AddJ.pt",  &_pt_jets);
	_output_tree->Branch("AddJ.pt_2", &_add_pt_test);
	_output_tree->Branch("AddJ.eta", &_eta_jets);
	_output_tree->Branch("AddJ.phi", &_phi_jets);
	_output_tree->Branch("AddJ.m",   &_m_jets);
	_output_tree->Branch("AddJ.N", &_n_add_j);

	_output_tree->Branch("Jet.N", &_n_jets);
	_output_tree->Branch("AddBJ.N", &_n_add_bjet);

//YY
	
	_output_tree->Branch("yy.HT_l1", &_Y_HT_l1);
	_output_tree->Branch("yy.HT_l2", &_Y_HT_l2);
	_output_tree->Branch("yy.HT_l3", &_Y_HT_l3);
	_output_tree->Branch("yy.HT_l4", &_Y_HT_l4);
	_output_tree->Branch("yy.HT_l5", &_Y_HT_l5);

	_output_tree->Branch("yy.HP_l1", &_Y_HP_l1);
	_output_tree->Branch("yy.HP_l2", &_Y_HP_l2);
	_output_tree->Branch("yy.HP_l3", &_Y_HP_l3);
	_output_tree->Branch("yy.HP_l4", &_Y_HP_l4);
	_output_tree->Branch("yy.HP_l5", &_Y_HP_l5);

	_output_tree->Branch("yy.HZ_l1", &_Y_HZ_l1);
	_output_tree->Branch("yy.HZ_l2", &_Y_HZ_l2);
	_output_tree->Branch("yy.HZ_l3", &_Y_HZ_l3);
	_output_tree->Branch("yy.HZ_l4", &_Y_HZ_l4);
	_output_tree->Branch("yy.HZ_l5", &_Y_HZ_l5);

	_output_tree->Branch("yy.HY_l1", &_Y_HY_l1);
	_output_tree->Branch("yy.HY_l2", &_Y_HY_l2);
	_output_tree->Branch("yy.HY_l3", &_Y_HY_l3);
	_output_tree->Branch("yy.HY_l4", &_Y_HY_l4);
	_output_tree->Branch("yy.HY_l5", &_Y_HY_l5);

	_output_tree->Branch("yy.HS_l1", &_Y_HS_l1);
	_output_tree->Branch("yy.HS_l2", &_Y_HS_l2);
	_output_tree->Branch("yy.HS_l3", &_Y_HS_l3);
	_output_tree->Branch("yy.HS_l4", &_Y_HS_l4);
	_output_tree->Branch("yy.HS_l5", &_Y_HS_l5);

	_output_tree->Branch("yy.H1_l1", &_Y_H1_l1);
	_output_tree->Branch("yy.H1_l2", &_Y_H1_l2);
	_output_tree->Branch("yy.H1_l3", &_Y_H1_l3);
	_output_tree->Branch("yy.H1_l4", &_Y_H1_l4);
	_output_tree->Branch("yy.H1_l5", &_Y_H1_l5);

//BB
	_output_tree->Branch("bb.HT_l1", &_B_HT_l1);
	_output_tree->Branch("bb.HT_l2", &_B_HT_l2);
	_output_tree->Branch("bb.HT_l3", &_B_HT_l3);
	_output_tree->Branch("bb.HT_l4", &_B_HT_l4);
	_output_tree->Branch("bb.HT_l5", &_B_HT_l5);

	_output_tree->Branch("bb.HP_l1", &_B_HP_l1);
	_output_tree->Branch("bb.HP_l2", &_B_HP_l2);
	_output_tree->Branch("bb.HP_l3", &_B_HP_l3);
	_output_tree->Branch("bb.HP_l4", &_B_HP_l4);
	_output_tree->Branch("bb.HP_l5", &_B_HP_l5);

	_output_tree->Branch("bb.HZ_l1", &_B_HZ_l1);
	_output_tree->Branch("bb.HZ_l2", &_B_HZ_l2);
	_output_tree->Branch("bb.HZ_l3", &_B_HZ_l3);
	_output_tree->Branch("bb.HZ_l4", &_B_HZ_l4);
	_output_tree->Branch("bb.HZ_l5", &_B_HZ_l5);

	_output_tree->Branch("bb.HY_l1", &_B_HY_l1);
	_output_tree->Branch("bb.HY_l2", &_B_HY_l2);
	_output_tree->Branch("bb.HY_l3", &_B_HY_l3);
	_output_tree->Branch("bb.HY_l4", &_B_HY_l4);
	_output_tree->Branch("bb.HY_l5", &_B_HY_l5);

	_output_tree->Branch("bb.HS_l1", &_B_HS_l1);
	_output_tree->Branch("bb.HS_l2", &_B_HS_l2);
	_output_tree->Branch("bb.HS_l3", &_B_HS_l3);
	_output_tree->Branch("bb.HS_l4", &_B_HS_l4);
	_output_tree->Branch("bb.HS_l5", &_B_HS_l5);

	_output_tree->Branch("bb.H1_l1", &_B_H1_l1);
	_output_tree->Branch("bb.H1_l2", &_B_H1_l2);
	_output_tree->Branch("bb.H1_l3", &_B_H1_l3);
	_output_tree->Branch("bb.H1_l4", &_B_H1_l4);
	_output_tree->Branch("bb.H1_l5", &_B_H1_l5);

//YYBB
	_output_tree->Branch("yybb.HT_l1", &_YB_HT_l1);
	_output_tree->Branch("yybb.HT_l2", &_YB_HT_l2);
	_output_tree->Branch("yybb.HT_l3", &_YB_HT_l3);
	_output_tree->Branch("yybb.HT_l4", &_YB_HT_l4);
	_output_tree->Branch("yybb.HT_l5", &_YB_HT_l5);

	_output_tree->Branch("yybb.HP_l1", &_YB_HP_l1);
	_output_tree->Branch("yybb.HP_l2", &_YB_HP_l2);
	_output_tree->Branch("yybb.HP_l3", &_YB_HP_l3);
	_output_tree->Branch("yybb.HP_l4", &_YB_HP_l4);
	_output_tree->Branch("yybb.HP_l5", &_YB_HP_l5);

	_output_tree->Branch("yybb.HZ_l1", &_YB_HZ_l1);
	_output_tree->Branch("yybb.HZ_l2", &_YB_HZ_l2);
	_output_tree->Branch("yybb.HZ_l3", &_YB_HZ_l3);
	_output_tree->Branch("yybb.HZ_l4", &_YB_HZ_l4);
	_output_tree->Branch("yybb.HZ_l5", &_YB_HZ_l5);

	_output_tree->Branch("yybb.HY_l1", &_YB_HY_l1);
	_output_tree->Branch("yybb.HY_l2", &_YB_HY_l2);
	_output_tree->Branch("yybb.HY_l3", &_YB_HY_l3);
	_output_tree->Branch("yybb.HY_l4", &_YB_HY_l4);
	_output_tree->Branch("yybb.HY_l5", &_YB_HY_l5);

	_output_tree->Branch("yybb.HS_l1", &_YB_HS_l1);
	_output_tree->Branch("yybb.HS_l2", &_YB_HS_l2);
	_output_tree->Branch("yybb.HS_l3", &_YB_HS_l3);
	_output_tree->Branch("yybb.HS_l4", &_YB_HS_l4);
	_output_tree->Branch("yybb.HS_l5", &_YB_HS_l5);

	_output_tree->Branch("yybb.H1_l1", &_YB_H1_l1);
	_output_tree->Branch("yybb.H1_l2", &_YB_H1_l2);
	_output_tree->Branch("yybb.H1_l3", &_YB_H1_l3);
	_output_tree->Branch("yybb.H1_l4", &_YB_H1_l4);
	_output_tree->Branch("yybb.H1_l5", &_YB_H1_l5);

//HH
	_output_tree->Branch("hh.HT_l1", &_HH_HT_l1);
	_output_tree->Branch("hh.HT_l2", &_HH_HT_l2);
	_output_tree->Branch("hh.HT_l3", &_HH_HT_l3);
	_output_tree->Branch("hh.HT_l4", &_HH_HT_l4);
	_output_tree->Branch("hh.HT_l5", &_HH_HT_l5);

	_output_tree->Branch("hh.HP_l1", &_HH_HP_l1);
	_output_tree->Branch("hh.HP_l2", &_HH_HP_l2);
	_output_tree->Branch("hh.HP_l3", &_HH_HP_l3);
	_output_tree->Branch("hh.HP_l4", &_HH_HP_l4);
	_output_tree->Branch("hh.HP_l5", &_HH_HP_l5);

	_output_tree->Branch("hh.HZ_l1", &_HH_HZ_l1);
	_output_tree->Branch("hh.HZ_l2", &_HH_HZ_l2);
	_output_tree->Branch("hh.HZ_l3", &_HH_HZ_l3);
	_output_tree->Branch("hh.HZ_l4", &_HH_HZ_l4);
	_output_tree->Branch("hh.HZ_l5", &_HH_HZ_l5);

	_output_tree->Branch("hh.HY_l1", &_HH_HY_l1);
	_output_tree->Branch("hh.HY_l2", &_HH_HY_l2);
	_output_tree->Branch("hh.HY_l3", &_HH_HY_l3);
	_output_tree->Branch("hh.HY_l4", &_HH_HY_l4);
	_output_tree->Branch("hh.HY_l5", &_HH_HY_l5);

	_output_tree->Branch("hh.HS_l1", &_HH_HS_l1);
	_output_tree->Branch("hh.HS_l2", &_HH_HS_l2);
	_output_tree->Branch("hh.HS_l3", &_HH_HS_l3);
	_output_tree->Branch("hh.HS_l4", &_HH_HS_l4);
	_output_tree->Branch("hh.HS_l5", &_HH_HS_l5);

	_output_tree->Branch("hh.H1_l1", &_HH_H1_l1);
	_output_tree->Branch("hh.H1_l2", &_HH_H1_l2);
	_output_tree->Branch("hh.H1_l3", &_HH_H1_l3);
	_output_tree->Branch("hh.H1_l4", &_HH_H1_l4);
	_output_tree->Branch("hh.H1_l5", &_HH_H1_l5);


	_output_tree->Branch("HT", &_H_T);
	_output_tree->Branch("HS", &_H_S);
	_output_tree->Branch("HZ", &_H_Z);
	_output_tree->Branch("HP", &_H_P);
	_output_tree->Branch("H1", &_H_1);
	_output_tree->Branch("HY", &_H_Y);
	_output_tree->Branch("L", &_L);

	_output_tree->Branch("met.px",     &_met_px);
	_output_tree->Branch("met.py",     &_met_py);
	_output_tree->Branch("met.e",      &_met_e);
	_output_tree->Branch("met.name",   &_met_name);
	
	_output_tree->Branch("met.TST",     &_met_TST);
	_output_tree->Branch("met.Soft",     &_met_Soft);	

	_output_tree->Branch("Event.BTag_D", &_btag_ds);

	_output_tree->Branch("Event.TotWeight",  &_event_total_weight);
	_output_tree->Branch("Event.Number",     &_event_number);
	_output_tree->Branch("Event.bbyyWeight", &_event_bbyy_weight);
	_output_tree->Branch("Event.Weight",     &_event_weight);
	_output_tree->Branch("Event.Xsec",       &_event_xsec);
	_output_tree->Branch("Event.Lumi",       &_event_lumi);
	_output_tree->Branch("Event.N",          &_nevent);
	_output_tree->Branch("Event.Cat",        &_event_cat);
	_output_tree->Branch("Event.BDT_Cat",    &_BDT_cat);
	_output_tree->Branch("Event.jet_rank",   &_jet_rank);
	
}
	





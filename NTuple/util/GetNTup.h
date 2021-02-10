#include "Variables.h"
#include "TSystemDirectory.h"
int GetNTuple(int i, int n, TString out){
	SetOuput(out,n);
	
	TChain *chain = Load();
	TChain *fc    = Load();
	xAOD::TEvent event( xAOD::TEvent::kBranchAccess );
	CHECK( event.readFrom( fc ) );
	SetBranches(chain);
	Long64_t nentries = chain->GetEntriesFast();
	_Thrust = GenerateT();
	std::cout<<"Total Number of Entries : "<<nentries<<std::endl;
	if( n == 0)
	{
		n=nentries;
	}
	std::cout<<"Event Will be processed : "<< n <<std::endl;
	for(unsigned int jentry = i; jentry<n; jentry++)
	{
		if(jentry%10000 == 0) std::cout<<" Events : "<<jentry<<"/"<<n<<std::endl;
		Int_t nb = chain->GetEntry(jentry);
		event.getEntry( jentry );
		const xAOD::EventInfo* ei = 0;
		CHECK( event.retrieve( ei, "EventInfo" ) );
		
		SetLumi(chain);	
		NewEvents();
		ClearQuantities();

		if(!Ana())
		{
			NewEvents();
			ClearQuantities();
			continue;
		}
		
		ComputeQuantities();
		_event_number = ei->eventNumber();
		_output4_tree->Fill();
		if(!_isData)
		{
		if(_event_number%4 == 0 || _event_number%4 == 1){
			_output1_tree->Fill();
		}else if(_event_number%4 == 2){
			_output2_tree->Fill();
		}else if(_event_number%4 == 3){
			_output3_tree->Fill();
		}
		}
		NewEvents();
		ClearQuantities();
		
		//std::cout<<"Loop finished"<<std::endl;
	}
	std::cout<<"Loop finished"<<std::endl;
	//_output_tree->SetDirectory(_output_file);
	//_output_tree->Write();
	_output1_tree->AutoSave();
	_output2_tree->AutoSave();
	_output3_tree->AutoSave();
	_output4_tree->AutoSave();
	_output_file->Close();
	return 1.;
}

TChain* Load()
{
	std::vector<TString> Samples;

	if(_isData)
	{
		for (unsigned int i = 0; i<_nSample; i++)
		{
		for (unsigned int j = 0; j<_nCompain; j++)
		{
			TString name = _GlobalPath;
			name += _Compain[j];
			name += "/";
			TSystemDirectory dir(name, name); 
			TList *files = dir.GetListOfFiles(); 
			if (files) { 
				TSystemFile *file; 
				TString fname; 
				TIter next(files); 
				while ((file=(TSystemFile*)next())) { 
					fname = file->GetName(); 
					if (!file->IsDirectory() && fname.EndsWith(".root")) { 
						Samples.push_back(name+"/"+fname); 
					} 
				} 
			}
		}
		}
	}else{
	for (unsigned int i = 0; i<_nSample; i++)
	{
		for (unsigned int j = 0; j<_nCompain; j++)
		{ 
			TString name = _GlobalPath;
			name += _Compain[j];
			name += "/Nominal/";
			name += _Compain[j];
			name += _Samplename[i];
			name += _Type;
			name += ".";
			name += _Tag[j];
			name += _Version;
			name += ".root";
			if(_isDirectory)
			{
				TSystemDirectory dir(name, name); 
				TList *files = dir.GetListOfFiles(); 
				if (files) { 
					TSystemFile *file; 
					TString fname; 
					TIter next(files); 
				while ((file=(TSystemFile*)next())) { 
					fname = file->GetName(); 
				if (!file->IsDirectory() && fname.EndsWith(".root")) { 
					Samples.push_back(name+"/"+fname); 
				} 
				} 
				}
			}else {
				Samples.push_back(name);
			}
		}
		
	}
	if(_isDirectory)
	{
		getNEventPerDirectory(Samples,_histo_name,_n_mc16a,_n_mc16d,_n_mc16e);
	}
	}

	
	std::cout<< TString::Format(" The total Nevent is : %.5f , %.5f, %.5f", _n_mc16a, _n_mc16d, _n_mc16e) <<std::endl;

	std::cout<<"Start Loading the Trees"<<std::endl;
	TChain *chain = GetChains(Samples, "CollectionTree");
	return chain;
}

TChain* GetChains(std::vector<TString> Samples, TString TreeName)
{
	TChain *chain = new TChain(TreeName);
	std::cout<<"Samples To Read "<<std::endl;
	for (unsigned int i = 0; i<Samples.size(); i++)
	{
		std::cout<<Samples[i]<<std::endl;
		chain->AddFile(Samples[i],-1,TreeName);
		//AddBranches(chain);
	}
	std::cout<<"Trees are loaded"<<std::endl;
	return chain;
}

bool Ana()
{
	
	if(!(_event_ispass)) return false;
	if(_event_N_lep != 0) return false; // Common selection
	if(!(_event_N_j_central < 6)) return false;
	if(_event_N_j_central < 2) return false;
	if(debug)std::cout<<"Event Passed"<<std::endl;
	for(unsigned int i=0; i<_gamma_pt->size(); i++)
	{
		TLorentzVector gamma;
		gamma.SetPtEtaPhiM(_gamma_pt->at(i),_gamma_eta->at(i),_gamma_phi->at(i),0.0);
		_photons.push_back(gamma);
	}
	if(_photons.size() != 2) return false;
	
	_Hyy = _photons.at(0) + _photons.at(1);

	if(_Hyy.M()*0.001 < 105 && _Hyy.M()*0.001 > 160) return false;

	if(debug)std::cout<<"Two Photons are selected"<<std::endl;

	CalibrateBJet();

	vector<TLorentzVector> bjets, nbjets;
	vector<Int_t> bjets_score;
	vector<Float_t> bjets_ptreco_sf;

	for(unsigned int i=0; i<_jets_pt->size(); i++)
	{
		Int_t rank = _jets_score->at(i); 
		TLorentzVector bjet, nbjet; 
		_jet_rank.push_back(rank);

		if(rank >= 2)
		{
			bjet.SetPtEtaPhiM(_jets_pt->at(i),_jets_eta->at(i),_jets_phi->at(i),_jets_m->at(i));
			bjets.push_back(bjet);
			bjets_score.push_back(_jets_score->at(i)-2);
			bjets_ptreco_sf.push_back(_PtReco_SF->at(i));
		}else{
			nbjet.SetPtEtaPhiM(_jets_pt->at(i),_jets_eta->at(i),_jets_phi->at(i),_jets_m->at(i));
			nbjets.push_back(nbjet);
		}
	}
	if(debug)std::cout<<"N B Jet : "<<bjets.size()<< " N Jet "<< bjets.size()+nbjets.size() << " NJet "<< _event_njets <<std::endl;

	int N77 = 0;
	for(unsigned int i = 0; i<_jets_pt->size(); i++)
	{
		if(_jets_btag_77->at(i)){N77++;}
	}
 	if(N77 >= 3) return false;
	//if(!(bjets.size() >= 2)) return false; //btag_85_77
	if(N77 != 2) return false;  //btag_77

	for(unsigned int j=1; j<=bjets_score.size(); j++)
	{
		for(unsigned int i=0; i<bjets_score.size()-1; i++)
		{
			if(bjets_score.at(i) < bjets_score.at(i+1))
			{
				int c;
				float sf;
				c  = bjets_score.at(i);
				sf = bjets_ptreco_sf.at(i);
				TLorentzVector jet(bjets.at(i));

				bjets_score.at(i) = bjets_score.at(i+1);
				bjets.at(i)       = bjets.at(i+1);
				bjets_ptreco_sf.at(i) = bjets_ptreco_sf.at(i+1);

				bjets_score.at(i+1) = c;
				bjets.at(i+1).SetPtEtaPhiM(jet.Pt(),jet.Eta(),jet.Phi(),jet.M());
				bjets_ptreco_sf.at(i+1) = sf;
			}
		} 
	}
	

	if(bjets_score.at(0) == bjets_score.at(1))
	{
		if(bjets.at(0).Pt() < bjets.at(1).Pt())
		{		
			int c;
			float sf;
			c  = bjets_score.at(0);
			sf = bjets_ptreco_sf.at(0);
			TLorentzVector jet(bjets.at(0));

			bjets_score.at(0) = bjets_score.at(1);
			bjets.at(0)       = bjets.at(1);
			bjets_ptreco_sf.at(0) = bjets_ptreco_sf.at(1);

			bjets_score.at(1) = c;
			bjets.at(1).SetPtEtaPhiM(jet.Pt(),jet.Eta(),jet.Phi(),jet.M());
			bjets_ptreco_sf.at(1) = sf;
		}
	}
	

	_Bjets.push_back(bjets.at(0));
	_Bjets.push_back(bjets.at(1));
	_score_b1 = bjets_score.at(0);
	_score_b2 = bjets_score.at(1);
	_sf_b1 = bjets_ptreco_sf.at(0);
	_sf_b2 = bjets_ptreco_sf.at(1);

	for(unsigned int i=2; i<bjets.size(); i++){_jets.push_back(bjets.at(i));}
	_n_add_bjet = _jets.size();
	for(unsigned int i=0; i<nbjets.size(); i++){_jets.push_back(nbjets.at(i));}

	if(debug){
	for(unsigned int i=0; i<bjets_score.size(); i++)
	{
		std::cout<<bjets_score.at(i)<<std::endl;
	}
	}
	if(debug)std::cout<<"Two BJets are selected"<<std::endl;

	_Hbb = _Bjets.at(0) + _Bjets.at(1);
	_HH  = _Hyy + _Hbb;
	for(unsigned int i=0; i<_jets.size(); i++) {_SumJ += _jets.at(i);}

	if(debug)std::cout<<" Mbb inv : "<<_Hbb.M()*0.001 <<std::endl;

	for(unsigned int i=0; i<_met_e->size(); i++)
	{
		//std::cout<<_met_name->at(i)<<std::endl;
		if(_met_name->at(i) == "TST")
		{
			_met_TST = _met_e->at(i);
		}
		if(_met_name->at(i) == "PVSoftTrk")
		{
			_met_Soft= _met_e->at(i);
		}
	}

	return true;
}

void CalibrateBJet()
{
	if(debug)std::cout<<"BJetCalibration is applied"<<std::endl;
	
	
	for(unsigned int i=0; i<_jets_pt->size(); i++)
	{
		TLorentzVector jet,mu;
		
		jet.SetPtEtaPhiM(_jets_pt->at(i),_jets_eta->at(i),_jets_phi->at(i),_jets_m->at(i));
		mu.SetPtEtaPhiM(_OneMu_pt->at(i),_OneMu_eta->at(i),_OneMu_phi->at(i),_OneMu_m->at(i));
		jet = (jet+mu)*_PtReco_SF->at(i);

		_jets_pt->at(i)  = jet.Pt();
		_jets_eta->at(i) = jet.Eta();
		_jets_phi->at(i) = jet.Phi();
		_jets_m->at(i)   = jet.M();
	}
	
	
	/*
	
	for(unsigned int i=0; i<_jets_pt->size(); i++)
	{
		TLorentzVector jet;
		
		jet.SetPtEtaPhiM(_jets_pt->at(i)*_Reg_SF->at(i),_jets_eta->at(i),_jets_phi->at(i),_jets_m->at(i));
		
		_jets_pt->at(i)  = jet.Pt();
		_jets_eta->at(i) = jet.Eta();
		_jets_phi->at(i) = jet.Phi();
		_jets_m->at(i)   = jet.M();
	}
	*/

}

void ComputeQuantities()
{
	
	_btag_ds = _score_b1-_score_b2;

	_pt_b1  = _Bjets.at(0).Pt();
	_pt_b2  = _Bjets.at(1).Pt();
	_eta_b1 = _Bjets.at(0).Eta();
	_eta_b2 = _Bjets.at(1).Eta();
	_phi_b1 = _Bjets.at(0).Phi();
	_phi_b2 = _Bjets.at(1).Phi();
	_e_b1   = _Bjets.at(0).E();
	_e_b2   = _Bjets.at(1).E();

	_pt_y1  = _photons.at(0).Pt();
	_pt_y2  = _photons.at(1).Pt();
	_eta_y1 = _photons.at(0).Eta();
	_eta_y2 = _photons.at(1).Eta();
	_phi_y1 = _photons.at(0).Phi();
	_phi_y2 = _photons.at(1).Phi();
	_e_y1   = _photons.at(0).E(); 
	_e_y2   = _photons.at(1).E();

	vector<TLorentzVector> _yybb, _hh;
	_yybb.push_back(_Bjets.at(0));
	_yybb.push_back(_Bjets.at(1));
	_yybb.push_back(_photons.at(0));
	_yybb.push_back(_photons.at(1));

	_hh.push_back(_Hyy);
	_hh.push_back(_Hbb);
	

	TVector3 thrust = GetThrust(_Bjets);
	FillLMomenta(_Bjets,20,thrust);

	_m_yy    = _Hyy.M();
	_m_bb    = _Hbb.M();
	_m_hh    = _HH.M();

	_pt_hh   = _HH.Pt();
	_pt_yy   = _Hyy.Pt();
	_pt_bb   = _Hbb.Pt();

	_eta_hh  = _HH.Eta();
	_eta_yy  = _Hyy.Eta();
	_eta_bb  = _Hbb.Eta();

	_phi_hh  = _HH.Phi();
	_phi_bb  = _Hbb.Phi();
	_phi_yy  = _Hyy.Phi();
	
	_e_hh    = _HH.E();
	_e_yy    = _Hyy.E();
	_e_bb    = _Hbb.E();

	_dr_hh    = _Hbb.DeltaR(_Hyy);
	_dr_bb    = _Bjets.at(0).DeltaR(_Bjets.at(1));
	_dr_yy    = _photons.at(0).DeltaR(_photons.at(1));
	_dr_b1yy  = _Hyy.DeltaR(_Bjets.at(0));
	_dr_b2yy  = _Hyy.DeltaR(_Bjets.at(1));
	_dr_y1bb  = _Hbb.DeltaR(_photons.at(0));
	_dr_y2bb  = _Hbb.DeltaR(_photons.at(1));

	_pt_jets  = _SumJ.Pt();
	_eta_jets = _SumJ.Eta();
	_phi_jets = _SumJ.Phi();
	_m_jets   = _SumJ.M();

	_n_add_j   = _jets.size()-_n_add_bjet;
	
	_n_jets    = _event_njets;

	FillFoxMomenta(_Bjets,20);

	_B_HT_l1 = FoxMomenta_l(_Bjets,'T',1);
	_B_HT_l2 = FoxMomenta_l(_Bjets,'T',2);
	_B_HT_l3 = FoxMomenta_l(_Bjets,'T',3);
	_B_HT_l4 = FoxMomenta_l(_Bjets,'T',4);
	_B_HT_l5 = FoxMomenta_l(_Bjets,'T',5);

	_B_HP_l1 = FoxMomenta_l(_Bjets,'P',1);
	_B_HP_l2 = FoxMomenta_l(_Bjets,'P',2);
	_B_HP_l3 = FoxMomenta_l(_Bjets,'P',3);
	_B_HP_l4 = FoxMomenta_l(_Bjets,'P',4);
	_B_HP_l5 = FoxMomenta_l(_Bjets,'P',5);

	_B_HS_l1 = FoxMomenta_l(_Bjets,'S',1);
	_B_HS_l2 = FoxMomenta_l(_Bjets,'S',2);
	_B_HS_l3 = FoxMomenta_l(_Bjets,'S',3);
	_B_HS_l4 = FoxMomenta_l(_Bjets,'S',4);
	_B_HS_l5 = FoxMomenta_l(_Bjets,'S',5);

	_B_HZ_l1 = FoxMomenta_l(_Bjets,'Z',1);
	_B_HZ_l2 = FoxMomenta_l(_Bjets,'Z',2);
	_B_HZ_l3 = FoxMomenta_l(_Bjets,'Z',3);
	_B_HZ_l4 = FoxMomenta_l(_Bjets,'Z',4);
	_B_HZ_l5 = FoxMomenta_l(_Bjets,'Z',5);

	_B_HY_l1 = FoxMomenta_l(_Bjets,'Y',1);
	_B_HY_l2 = FoxMomenta_l(_Bjets,'Y',2);
	_B_HY_l3 = FoxMomenta_l(_Bjets,'Y',3);
	_B_HY_l4 = FoxMomenta_l(_Bjets,'Y',4);
	_B_HY_l5 = FoxMomenta_l(_Bjets,'Y',5);

	_B_H1_l1 = FoxMomenta_l(_Bjets,'1',1);
	_B_H1_l2 = FoxMomenta_l(_Bjets,'1',2);
	_B_H1_l3 = FoxMomenta_l(_Bjets,'1',3);
	_B_H1_l4 = FoxMomenta_l(_Bjets,'1',4);
	_B_H1_l5 = FoxMomenta_l(_Bjets,'1',5);

//PHOTONS

	_Y_HT_l1 = FoxMomenta_l(_photons,'T',1);
	_Y_HT_l2 = FoxMomenta_l(_photons,'T',2);
	_Y_HT_l3 = FoxMomenta_l(_photons,'T',3);
	_Y_HT_l4 = FoxMomenta_l(_photons,'T',4);
	_Y_HT_l5 = FoxMomenta_l(_photons,'T',5);

	_Y_HP_l1 = FoxMomenta_l(_photons,'P',1);
	_Y_HP_l2 = FoxMomenta_l(_photons,'P',2);
	_Y_HP_l3 = FoxMomenta_l(_photons,'P',3);
	_Y_HP_l4 = FoxMomenta_l(_photons,'P',4);
	_Y_HP_l5 = FoxMomenta_l(_photons,'P',5);

	_Y_HS_l1 = FoxMomenta_l(_photons,'S',1);
	_Y_HS_l2 = FoxMomenta_l(_photons,'S',2);
	_Y_HS_l3 = FoxMomenta_l(_photons,'S',3);
	_Y_HS_l4 = FoxMomenta_l(_photons,'S',4);
	_Y_HS_l5 = FoxMomenta_l(_photons,'S',5);

	_Y_HZ_l1 = FoxMomenta_l(_photons,'Z',1);
	_Y_HZ_l2 = FoxMomenta_l(_photons,'Z',2);
	_Y_HZ_l3 = FoxMomenta_l(_photons,'Z',3);
	_Y_HZ_l4 = FoxMomenta_l(_photons,'Z',4);
	_Y_HZ_l5 = FoxMomenta_l(_photons,'Z',5);

	_Y_HY_l1 = FoxMomenta_l(_photons,'Y',1);
	_Y_HY_l2 = FoxMomenta_l(_photons,'Y',2);
	_Y_HY_l3 = FoxMomenta_l(_photons,'Y',3);
	_Y_HY_l4 = FoxMomenta_l(_photons,'Y',4);
	_Y_HY_l5 = FoxMomenta_l(_photons,'Y',5);

	_Y_H1_l1 = FoxMomenta_l(_photons,'1',1);
	_Y_H1_l2 = FoxMomenta_l(_photons,'1',2);
	_Y_H1_l3 = FoxMomenta_l(_photons,'1',3);
	_Y_H1_l4 = FoxMomenta_l(_photons,'1',4);
	_Y_H1_l5 = FoxMomenta_l(_photons,'1',5);

//YY+BB


	_YB_HT_l1 = FoxMomenta_l(_yybb,'T',1);
	_YB_HT_l2 = FoxMomenta_l(_yybb,'T',2);
	_YB_HT_l3 = FoxMomenta_l(_yybb,'T',3);
	_YB_HT_l4 = FoxMomenta_l(_yybb,'T',4);
	_YB_HT_l5 = FoxMomenta_l(_yybb,'T',5);

	_YB_HP_l1 = FoxMomenta_l(_yybb,'P',1);
	_YB_HP_l2 = FoxMomenta_l(_yybb,'P',2);
	_YB_HP_l3 = FoxMomenta_l(_yybb,'P',3);
	_YB_HP_l4 = FoxMomenta_l(_yybb,'P',4);
	_YB_HP_l5 = FoxMomenta_l(_yybb,'P',5);

	_YB_HS_l1 = FoxMomenta_l(_yybb,'S',1);
	_YB_HS_l2 = FoxMomenta_l(_yybb,'S',2);
	_YB_HS_l3 = FoxMomenta_l(_yybb,'S',3);
	_YB_HS_l4 = FoxMomenta_l(_yybb,'S',4);
	_YB_HS_l5 = FoxMomenta_l(_yybb,'S',5);

	_YB_HZ_l1 = FoxMomenta_l(_yybb,'Z',1);
	_YB_HZ_l2 = FoxMomenta_l(_yybb,'Z',2);
	_YB_HZ_l3 = FoxMomenta_l(_yybb,'Z',3);
	_YB_HZ_l4 = FoxMomenta_l(_yybb,'Z',4);
	_YB_HZ_l5 = FoxMomenta_l(_yybb,'Z',5);

	_YB_HY_l1 = FoxMomenta_l(_yybb,'Y',1);
	_YB_HY_l2 = FoxMomenta_l(_yybb,'Y',2);
	_YB_HY_l3 = FoxMomenta_l(_yybb,'Y',3);
	_YB_HY_l4 = FoxMomenta_l(_yybb,'Y',4);
	_YB_HY_l5 = FoxMomenta_l(_yybb,'Y',5);

	_YB_H1_l1 = FoxMomenta_l(_yybb,'1',1);
	_YB_H1_l2 = FoxMomenta_l(_yybb,'1',2);
	_YB_H1_l3 = FoxMomenta_l(_yybb,'1',3);
	_YB_H1_l4 = FoxMomenta_l(_yybb,'1',4);
	_YB_H1_l5 = FoxMomenta_l(_yybb,'1',5);

// HH 

	_HH_HT_l1 = FoxMomenta_l(_hh,'T',1);
	_HH_HT_l2 = FoxMomenta_l(_hh,'T',2);
	_HH_HT_l3 = FoxMomenta_l(_hh,'T',3);
	_HH_HT_l4 = FoxMomenta_l(_hh,'T',4);
	_HH_HT_l5 = FoxMomenta_l(_hh,'T',5);

	_HH_HP_l1 = FoxMomenta_l(_hh,'P',1);
	_HH_HP_l2 = FoxMomenta_l(_hh,'P',2);
	_HH_HP_l3 = FoxMomenta_l(_hh,'P',3);
	_HH_HP_l4 = FoxMomenta_l(_hh,'P',4);
	_HH_HP_l5 = FoxMomenta_l(_hh,'P',5);

	_HH_HS_l1 = FoxMomenta_l(_hh,'S',1);
	_HH_HS_l2 = FoxMomenta_l(_hh,'S',2);
	_HH_HS_l3 = FoxMomenta_l(_hh,'S',3);
	_HH_HS_l4 = FoxMomenta_l(_hh,'S',4);
	_HH_HS_l5 = FoxMomenta_l(_hh,'S',5);

	_HH_HZ_l1 = FoxMomenta_l(_hh,'Z',1);
	_HH_HZ_l2 = FoxMomenta_l(_hh,'Z',2);
	_HH_HZ_l3 = FoxMomenta_l(_hh,'Z',3);
	_HH_HZ_l4 = FoxMomenta_l(_hh,'Z',4);
	_HH_HZ_l5 = FoxMomenta_l(_hh,'Z',5);

	_HH_HY_l1 = FoxMomenta_l(_hh,'Y',1);
	_HH_HY_l2 = FoxMomenta_l(_hh,'Y',2);
	_HH_HY_l3 = FoxMomenta_l(_hh,'Y',3);
	_HH_HY_l4 = FoxMomenta_l(_hh,'Y',4);
	_HH_HY_l5 = FoxMomenta_l(_hh,'Y',5);

	_HH_H1_l1 = FoxMomenta_l(_hh,'1',1);
	_HH_H1_l2 = FoxMomenta_l(_hh,'1',2);
	_HH_H1_l3 = FoxMomenta_l(_hh,'1',3);
	_HH_H1_l4 = FoxMomenta_l(_hh,'1',4);
	_HH_H1_l5 = FoxMomenta_l(_hh,'1',5);


}

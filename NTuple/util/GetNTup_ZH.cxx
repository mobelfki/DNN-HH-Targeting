#include "Variables_H7_ZH.h"
#include "GetNTup.h"

int main(int argc, char* argv []){

	if(argc == 1)
	{
		std::cout<<" Please set number of event : 0 no limit"<<std::endl;	
		return 0;
	}
	if(argc == 4)
	{
		std::cout<<" splitted : To splite your output"<<std::endl;
	}
	const char* APP_NAME = argv[ 0 ];
	CHECK( xAOD::Init( APP_NAME ) ); 
	int i,n;
	TString out;
	if(argc == 4)
	{
		i = std::stoi(argv[1]);
		n = std::stoi(argv[2]);
		out = argv[3];
	}
	
	if(argc == 2)
	{
		i = 0;
		n = std::stoi(argv[1]);
		out = "";
	}else if(argc == 3){

		i = std::stoi(argv[1]);
		n = std::stoi(argv[2]);
		out = "";
	}
	GetNTuple(i,n,argv[3]);
	return 0;
}

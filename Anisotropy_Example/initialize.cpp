#ifndef SOLIDIFICATION_INIT
#define SOLIDIFICATION_INIT
#include<iomanip>
#include<cmath>
#include<ctime>
#include<iostream>
#include<fstream>
#include<sstream>
#include<cstdlib>
#include<cctype>
#include<time.h>
#include"MMSP.hpp"
#include"solidification.hpp"

namespace MMSP
{

void generate(int dim, const char* filename)
{
	const int L=500;
	const int L2 = 250;
	const double deltaX=0.025;
	const double undercooling=-0.5;
	if (dim==2) {
		GRID2D initGrid(2,0,L,0,L2);
		for (int d=0; d<dim; ++d) dx(initGrid,d)=deltaX;

		// Seed a circle of radius N*dx
		int R=5;
		for (int i=0; i<nodes(initGrid); ++i) {
			initGrid(i)[1]=undercooling; // Initial undercooling
			vector<int> x = position(initGrid,i);
			int r=sqrt(pow(x[0],2)+pow(x[1],2));
			if (r<=R) initGrid(i)[0]=1.;
			else initGrid(i)[0]=0.;
		}
		output(initGrid,filename);
	} else {
		std::cerr<<"Anisotropic solidification code is only implemented for 2D."<<std::endl;
		MMSP::Abort(-1);
	}
}
}//namespace MMSP
#endif
int main(int argc, char* argv[]){
	MMSP::Init(argc,argv);
	
	int dim=2; //model only implemented for 2D
	std::string outfile;
	if (argc==2){
		outfile=argv[1];
	}
	else outfile="solidification_data";

	char* filename = new char[outfile.length()+1];
	for (unsigned int i=0; i<outfile.length(); i++)
		filename[i] = outfile[i];
	filename[outfile.length()]='\0';

	// generate initial grid
	MMSP::generate(dim, filename);

	delete [] filename;

}
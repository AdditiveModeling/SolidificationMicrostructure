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
	const int L = 100;
	const double deltaX = 4.6e-6; //grid spacing, cm
	const double c_inf = 0.40831;

	if (dim == 2) {
		GRID2D initGrid(2,0,L,0,L);
		for (int d = 0; d < dim; d++)
			dx(initGrid,d) = deltaX;
		
		// Seed a diamond of size diamondParam
		int diamondParam = 200;
		for (int i = 0; i < nodes(initGrid); i++) {
			initGrid(i)[1] = c_inf; // Initial concentration
			vector<int> x = position(initGrid,i);
			int r = abs(x[0]-L/2) + abs(x[1]-L/2);
			if (r <= diamondParam)
				initGrid(i)[0] = 0.;
			else
				initGrid(i)[0] = 1.;
		}
		
		output(initGrid,filename);
	} else {
		std::cerr << "Anisotropic solidification code is only implemented for 2D." << std::endl;
		MMSP::Abort(-1);
	}
} 
}//namespace MMSP
#endif
int main(int argc, char* argv[]){
	MMSP::Init(argc,argv);
	
	int dim; 
	std::string outfile;
	if (argc==3){
		outfile=argv[1];
		dim=atoi(argv[2]);
	}
	else{
		outfile="solidification_data";
		dim=atoi(argv[1]);
	}

	char* filename = new char[outfile.length()+1];
	for (unsigned int i=0; i<outfile.length(); i++)
		filename[i] = outfile[i];
	filename[outfile.length()]='\0';

	// generate initial grid
	MMSP::generate(dim, filename);

	delete [] filename;

}

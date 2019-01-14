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

//global dimension. Must be defined, else the sim won't generate!
int dim = 0;

//global int defining the mode. default zero
//0: default, single circular/spherical seed at center
//1: multiple random
int mode = 0; 

//global int for number of initial grains in mode 1, default 10
int grains = 10;

//global quartet of doubles defining the orientation, default 1 0 0 0
//in mode 0 this will be the orientation everywhere, in mode 1 this will be the initial liquid orientation
//quaternion will be normalized automatically if magnitude != 1
//only _q1 and _q4 are used in 2D simulations (complex number orientation)
double _q1 = 1.;
double _q2 = 0.;
double _q3 = 0.;
double _q4 = 0.;

//global initial concentration value, default 0.40831
double _c = 0.40831;

//global sim dimensions, default values 0. Will be overwritten to true defaults if these remain 0
int L1 = 0;
int L2 = 0;
int L3 = 0;

namespace MMSP
{
	void generate(int dim, const char* filename)
	{	
		if (dim==2){
			if(L2 == 0) {
				L1 = 400;
				L2 = 400;
			}
			const double deltaX=0.0000046;
			
			GRID2D initGrid(4,0,L1,0,L2);
			for (int d=0; d<dim; ++d) dx(initGrid,d)=deltaX;
			
			// Seed a circle of radius N*dx
			int R=2;
			for (int i=0; i<nodes(initGrid); ++i) {
				initGrid(i)[1]=_c; // Initial composition
				initGrid(i)[2]=_q1;
				initGrid(i)[3]=_q4;
				vector<int> x = position(initGrid,i);
				double r=pow(x[0]-L1/2,2)+pow(x[1]-L2/2,2);
				if (r<R*R) {
					initGrid(i)[0]=1.;
				} 
				else {
					initGrid(i)[0]=0.;
				}
			}
			output(initGrid,filename);
			
		}
		if (dim==3) {
			if(L3 == 0) {
				L1 = 200;
				L2 = 200;
				L3 = 200;
			}
			const double deltaX=0.0000046;
			GRID3D initGrid(6,0,L1,0,L2,0,L3);
			for (int d=0; d<dim; ++d) dx(initGrid,d)=deltaX;
			
			// Seed a circle of radius N*dx
			int R=5;
			for (int i=0; i<nodes(initGrid); ++i) {
				initGrid(i)[1]=_c; // Initial composition
				initGrid(i)[2]=_q1;
				initGrid(i)[3]=_q2;
				initGrid(i)[4]=_q3;
				initGrid(i)[5]=_q4;
				vector<int> x = position(initGrid,i);
				int r=sqrt(pow(x[0]-L1/2,2)+pow(x[1]-L2/2,2)+pow(x[2]-L3/2,2));
				if (r<=R) {
					initGrid(i)[0]=1.;
				} 
				else {
					initGrid(i)[0]=0.;
				}
			}
			output(initGrid,filename);
			
		}
		
	} 

}//namespace MMSP
#endif
int main(int argc, char* argv[]){
	MMSP::Init(argc,argv);
	
	std::string outfile = "grid";
	if (argc == 1) {
		std::cout << PROGRAM << ": " << MESSAGE << "\n\n";
		std::cout << "Valid command lines have the form:\n\n";
		std::cout << "    " << PROGRAM << " ";
		std::cout << "[--help] [arguments]\n\n";
		std::cout << "Arguments are of the form: _:_____\n";
		std::cout << "The first letter represents the field to be given a value\n";
		std::cout << "The second part is the value you wish to submit\n";
		std::cout << "The following fields are available: \n";
		std::cout << "d: dimension. submit an integer for how many dimensions you want the sim to be\n";
		std::cout << "o: orientation. submit 4 doubles separated by commas (e.g.: \"o:0.5,0.5,0.5,0.5\" \n)";
		std::cout << "m: mode. Type of simulation done, 0 for single grain, 1 for multiple random grains\n";
		std::cout << "g: grains. Integer number of grains you want for mode 1\n";
		std::cout << "l: lengths. Submit 2 or 3 integers to define what the dimensions of the simulation region are\n";
		std::cout << "c: composition. Submit double between 0 and 1 to define the starting chemical composition\n\t(0: pure nickel, 1: pure copper)\n";
		std::cout << "f: filename. Give the output file a name. Default is \"grid\"";
		exit(0);
	}
	else if (std::string(argv[1]) == std::string("--help")) {
		std::cout << PROGRAM << ": " << MESSAGE << "\n\n";
		std::cout << "Valid command lines have the form:\n\n";
		std::cout << "    " << PROGRAM << " ";
		std::cout << "[--help] [arguments]\n\n";
		std::cout << "Arguments are of the form: _:_____\n";
		std::cout << "The first letter represents the field to be given a value\n";
		std::cout << "The second part is the value you wish to submit\n";
		std::cout << "The following fields are available: \n";
		std::cout << "d: dimension. submit an integer for how many dimensions you want the sim to be\n";
		std::cout << "o: orientation. submit 4 doubles separated by commas (e.g.: \"o:0.5,0.5,0.5,0.5\" \n)";
		std::cout << "m: mode. Type of simulation done, 0 for single grain, 1 for multiple random grains\n";
		std::cout << "g: grains. Integer number of grains you want for mode 1\n";
		std::cout << "l: lengths. Submit 2 or 3 integers to define the sim region dimensions, separated by commas\n";
		std::cout << "c: composition. Submit double between 0 and 1 to define the starting chemical composition\n\t(0: pure nickel, 1: pure copper)\n";
		exit(0);
	}
	else {
		for(int i = 1; i < argc; i++) {
			if(argv[i][0] == 'd') {
				dim = atoi(argv[i]+2);
			}
			else if(argv[i][0] == 'o') {
				char* token = strtok(argv[i]+2, ",");
				_q1 = atof(token);
				token = strtok(NULL, ",");
				_q2 = atof(token);
				token = strtok(NULL, ",");
				_q3 = atof(token);
				token = strtok(NULL, ",");
				_q4 = atof(token);
			}
			else if(argv[i][0] == 'm') {
				mode = atoi(argv[i]+2);
			}
			else if(argv[i][0] == 'g') {
				grains = atoi(argv[i]+2);
			}
			else if(argv[i][0] == 'l') {
				char* token = strtok(argv[i]+2, ",");
				L1 = atoi(token);
				token = strtok(NULL, ",");
				L2 = atoi(token);
				token = strtok(NULL, ",");
				if(token != NULL) {
					L3 = atoi(token);
				}
			}
			else if(argv[i][0] == 'c') {
				_c = atof(argv[i]+2);
			}
			else if(argv[i][0] == 'f') {
				outfile = argv[i]+2;
			}
		}
	}

	char* filename = new char[outfile.length()+1];
	for (unsigned int i=0; i<outfile.length(); i++)
		filename[i] = outfile[i];
	filename[outfile.length()]='\0';

	// generate initial grid
	MMSP::generate(dim, filename);

	delete [] filename;

}
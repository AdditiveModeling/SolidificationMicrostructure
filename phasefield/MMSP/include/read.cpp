#ifndef SOLIDIFICATION_READ
#define SOLIDIFICATION_READ
#include<iomanip>
#include<cmath>
#include<ctime>
#include<iostream>
#include<fstream>
#include<sstream>
#include<cstdlib>
#include<cctype>
#include<time.h>


namespace MMSP{
int read_dim(int argc, char* argv[]){
	// file open error check
	std::ifstream input(argv[1]);
	if (!input) {
		std::cerr << "File input error: could not open " << argv[1] << ".\n\n";
		MMSP::Abort(-1);
	}

	// read data type
	std::string type;
	getline(input, type, '\n');

	// grid type error check
	if (type.substr(0, 4) != "grid") {
		std::cerr << "File input error: file does not contain grid data." << std::endl;
		MMSP::Abort(-1);
	}

	// read grid dimension
	int dim;
	input >> dim;

	input.close();


	return dim;
}
} //namespace MMSP
#endif
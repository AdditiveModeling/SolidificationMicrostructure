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
	MMSP::vector<double> read_T(const char* T_txt){
		std::ifstream T_file;
		double x;
		long count=0;
		double step_size=0.;
		MMSP::vector<double> T_vector;
		T_file.open(T_txt);
		if (!T_file){
			std::cout<< "Unable to open temperature file";
			exit(1);}
		while (T_file>>x){
			if (count==0){
				step_size=x;
				count+=1;
			}
			else{
				T_vector.append(x);
			}
		}
		T_file.close();
		return T_vector;
}
		double read_step_size(const char* T_txt){
			std::ifstream T_file;
			double x;
			long count=0;
			double step_size=0.;
			MMSP::vector<double> T_vector;
			T_file.open(T_txt);
			if (!T_file){
				std::cout<< "Unable to open temperature file";
				exit(1);}
			while (T_file>>x){
				if (count==0){
					step_size=x;
					break;}
			}
			return step_size;
		}

} //namespace MMSP
#endif
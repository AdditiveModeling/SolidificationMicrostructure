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
		
		double step_size=0.;
		MMSP::vector<double> T_vector;
		T_file.open(T_txt);
		if (!T_file){
			std::cout<< "Unable to open temperature file";
			exit(1);}
		double temperature_value;
		int loc;
		std::string temperature_string;
		std::string line_string;
		while (getline(T_file,line_string,',')){
			loc=line_string.find_first_of(',');
			temperature_string=line_string.substr(loc+1);
			char temperature_char[255];
			strcpy(temperature_char, temperature_string.c_str());
			temperature_value=atof(temperature_char);
			T_vector.append(temperature_value);
			}
		
		T_file.close();
		
		return T_vector;


}
double read_step_size(const char* T_csv){
		std::ifstream T_file;
		double x;
		long count=0;
		double step_size;
		MMSP::vector<double> T_vector;
		T_file.open(T_csv);
		if (!T_file){
			std::cout<< "Unable to open temperature file";
			exit(1);}
		std::string step_string;
		std::string line_string;
		int loc;
		while (getline(T_file,line_string)){
			loc=line_string.find_first_of(',');
			step_string=line_string.substr(0,loc);
			char step_char[255];
			strcpy(step_char, step_string.c_str());
			if (count==1){
				step_size=atof(step_char);
				break;
			}
			count+=1;
		}
		
		return step_size;
	}

} //namespace MMSP
#endif
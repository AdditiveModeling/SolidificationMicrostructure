#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cctype>
#include <vector>
#include <time.h>
#include "MMSP.hpp"
#include "read.cpp"
#include "spline.h"

int main(){
	double step_dt=MMSP::read_step_size("T.csv");
	MMSP::vector<double> T_vec_MMSP=MMSP::read_T("T.csv");
	std::vector<double> T_vec= std::vector<double>(T_vec_MMSP.length());
	for (int m; m<T_vec.size(); m++){
		T_vec[m]=T_vec_MMSP[m];
	}
	std::vector<double> time_vec= std::vector<double>(T_vec.size());
	for (int n; n<time_vec.size(); n++){
		time_vec[n]=n*step_dt;
	}
	double dt_find=step_dt;

	const double original_step=step_dt;

	std::vector<double> new_T(2*T_vec.size());

	tk::spline s;
	s.set_points(time_vec,T_vec);

	for (int k=0; k<new_T.size()-1; k++){
		if (k%2==0){
			new_T[k]=T_vec[k/2];
		}
		else{
			new_T[k]=s(time_vec[(k-1)/2]+step_dt*0.5);
		}
	}
	dt_find/=2.;
	step_dt=dt_find;
	std::ofstream new_T_file;
	new_T_file.open("new_T.csv");
	for (int j=0; j<new_T.size()-1; j++){
		new_T_file<<j*step_dt<<","<<new_T[j]<<"\n";
	}
	return 0;
}




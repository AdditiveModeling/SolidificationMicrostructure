#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cctype>
#include <time.h>
#include "MMSP.hpp"
#include "solidification.hpp"
#include "read.cpp"
#include "update.cpp"

void print(std::string stuff) {
	std::cout << stuff << "\n";
}

int main(int argc, char* argv[]){
	#ifdef MPI_VERSION
	MPI::Init();
	#endif
	int dim=MMSP::read_dim(argc,argv);
	

	// bad argument list
	if (argc<3 or argc>5) {
		std::cout << PROGRAM << ": bad argument list.  Use\n\n";
		std::cout << "    " << PROGRAM << " --help\n\n";
		std::cout << "to generate help message.\n\n";
		MMSP::Abort(-1);
	}

	int steps;
	int increment;
	std::string outfile;

	if (std::string(argv[2]).find_first_not_of("0123456789") == std::string::npos) {
		// set output file name
		outfile = argv[1];

		// must have integral number of time steps
		if (std::string(argv[2]).find_first_not_of("0123456789") != std::string::npos) {
			std::cout << PROGRAM << ": number of time steps must have integral value.  Use\n\n";
			std::cout << "    " << PROGRAM << " --help\n\n";
			std::cout << "to generate help message.\n\n";
			MMSP::Abort(-1);
		}

		steps = atoi(argv[2]);
		increment = steps;

		if (argc > 3) {
			// must have integral output increment
			if (std::string(argv[3]).find_first_not_of("0123456789") != std::string::npos) {
				std::cout << PROGRAM << ": output increment must have integral value.  Use\n\n";
				std::cout << "    " << PROGRAM << " --help\n\n";
				std::cout << "to generate help message.\n\n";
				MMSP::Abort(-1);
			}

			increment = atoi(argv[3]);

			// output increment must be smaller than number of steps
			if (increment > steps) {
				std::cout << PROGRAM << ": output increment must be smaller than number of time steps.  Use\n\n";
				std::cout << "    " << PROGRAM << " --help\n\n";
				std::cout << "to generate help message.\n\n";
				MMSP::Abort(-1);
			}
		}
	}

	else {
		// set output file name
		outfile = argv[2];

		// set number of time steps
		if (std::string(argv[3]).find_first_not_of("0123456789") != std::string::npos) {
			// must have integral number of time steps
			std::cout << PROGRAM << ": number of time steps must have integral value.  Use\n\n";
			std::cout << "    " << PROGRAM << " --help\n\n";
			std::cout << "to generate help message.\n\n";
			MMSP::Abort(-1);
		}

		steps = atoi(argv[3]);
		increment = steps;

		if (argc > 4) {
			// must have integral output increment
			if (std::string(argv[4]).find_first_not_of("0123456789") != std::string::npos) {
				std::cout << PROGRAM << ": output increment must have integral value.  Use\n\n";
				std::cout << "    " << PROGRAM << " --help\n\n";
				std::cout << "to generate help message.\n\n";
				MMSP::Abort(-1);
			}

			increment = atoi(argv[4]);

			// output increment must be smaller than number of steps
			if (increment > steps) {
				std::cout << PROGRAM << ": output increment must be smaller than number of time steps.  Use\n\n";
				std::cout << "    " << PROGRAM << " --help\n\n";
				std::cout << "to generate help message.\n\n";
				MMSP::Abort(-1);
			}
		}
	}


	// set output file basename
	int iterations_start(0);
	if (outfile.find_first_of(".") != outfile.find_last_of(".")) {
		std::string number = outfile.substr(outfile.find_first_of(".") + 1, outfile.find_last_of(".") - 1);
		iterations_start = atoi(number.c_str());
	}
	std::string base;
	if (outfile.find(".", outfile.find_first_of(".") + 1) == std::string::npos) // only one dot found
		base = outfile.substr(0, outfile.find_last_of(".")) + ".";
	else {
		int last_dot = outfile.find_last_of(".");
		int prev_dot = outfile.rfind('.', last_dot - 1);
		std::string number = outfile.substr(prev_dot + 1, last_dot - prev_dot - 1);
		bool isNumeric(true);
		for (unsigned int i = 0; i < number.size(); ++i) {
			if (!isdigit(number[i])) isNumeric = false;
		}
		if (isNumeric)
			base = outfile.substr(0, outfile.rfind(".", outfile.find_last_of(".") - 1)) + ".";
		else base = outfile.substr(0, outfile.find_last_of(".")) + ".";
	}

	// set output file suffix
	std::string suffix = "";
	if (outfile.find_last_of(".") != std::string::npos)
		suffix = outfile.substr(outfile.find_last_of("."), std::string::npos);

	// set output filename length
	int length = base.length() + suffix.length();
	if (1) {
		std::stringstream slength;
		slength << steps;
		length += slength.str().length();
	}
	
	// perform computation
	if (dim == 1) {
			// construct grid object
			GRID1D grid(argv[1]);

			// perform computation
			for (int i = iterations_start; i < steps; i += increment) {
				MMSP::update(grid, increment);

				// generate output filename
				std::stringstream outstr;
				int n = outstr.str().length();
				for (int j = 0; n < length; j++) {
					outstr.str("");
					outstr << base;
					for (int k = 0; k < j; k++) outstr << 0;
					outstr << i + increment << suffix;
					n = outstr.str().length();
				}

				char filename[FILENAME_MAX] = {}; // initialize null characters
				for (unsigned int j=0; j<outstr.str().length(); j++)
					filename[j]=outstr.str()[j];

				// write grid output to file
				MMSP::output(grid, filename);
			}
		} else if (dim == 2) {
			// construct grid object
			GRID2D grid(argv[1]);
			//MMSP::vector<int> sampler(2,0);
			//std::cout << grid(sampler)[2]<< grid(sampler)[3] <<"\n";

			// perform computation
			for (int i = iterations_start; i < steps; i += increment) {
				MMSP::update(grid, increment);

				// generate output filename
				std::stringstream outstr;
				int n = outstr.str().length();
				for (int j = 0; n < length; j++) {
					outstr.str("");
					outstr << base;
					for (int k = 0; k < j; k++) outstr << 0;
					outstr << i + increment << suffix;
					n = outstr.str().length();
				}

				char filename[FILENAME_MAX] = {}; // initialize null characters
				for (unsigned int j=0; j<outstr.str().length(); j++)
					filename[j]=outstr.str()[j];

				// write grid output to file
				MMSP::output(grid, filename);
			}
		} else if (dim == 3) {
			// construct grid object
			GRID3D grid(argv[1]);

			// perform computation
			for (int i = iterations_start; i < steps; i += increment) {
				MMSP::update(grid, increment);
				// generate output filename
				std::stringstream outstr;
				int n = outstr.str().length();
				for (int j = 0; n < length; j++) {
					outstr.str("");
					outstr << base;
					for (int k = 0; k < j; k++) outstr << 0;
					outstr << i + increment << suffix;
					n = outstr.str().length();
				}
				char filename[FILENAME_MAX] = {}; // initialize null characters
				for (unsigned int j=0; j<outstr.str().length(); j++)
					filename[j]=outstr.str()[j];

				// write grid output to file
				MMSP::output(grid, filename);
			}
		}

		MMSP::Finalize();
}

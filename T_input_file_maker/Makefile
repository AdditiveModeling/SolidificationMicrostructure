# includes
incdir = $(MMSP_PATH)/include

# compilers/flags
compiler = g++ -O3
flags = -I $(incdir)

# the program
smooth: smooth.cpp
	$(compiler) $(flags) $< -o $@.out -lz


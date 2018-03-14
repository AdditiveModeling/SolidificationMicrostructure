# Makefile
# GNU makefile for dendritic solidification code
# Questions/comments to trevor.keller@gmail.com (Trevor Keller)

# includes
incdir = $(MMSP_PATH)/include

# compilers/flags
compiler = g++ -O3
pcompiler = mpic++ -O3
flags = -I $(incdir)
pflags = $(flags) -include mpi.h

# the program
initialize: initialize.cpp
	$(compiler) $(flags) $< -o $@ -lz

simulate: simulate.cpp
	$(compiler) $(flags) $< -o $@ -lz

parallel: simulate.cpp
	$(pcompiler) $(pflags) $< -o $@ -lz

clean:
	rm -rf simulate initialize parallel *~ *.dat *.vti

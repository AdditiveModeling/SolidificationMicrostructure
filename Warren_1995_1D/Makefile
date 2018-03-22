# Makefile
# GNU makefile for dendritic solidification code
# Questions/comments to trevor.keller@gmail.com (Trevor Keller)

# includes
incdir = $(CURDIR)/include

# compilers/flags
compiler = g++ -O3
pcompiler = mpic++ -O3
flags = -I $(incdir)
pflags = $(flags) -include mpi.h

# the program
initialize: initialize.cpp
	$(compiler) $(flags) $< -o $@.out -lz

simulate: simulate.cpp
	$(compiler) $(flags) $< -o $@.out -lz

parallel: simulate.cpp
	$(pcompiler) $(pflags) $< -o $@.out -lz

clean:
	rm -rf simulate.out initialize.out parallel.out *~ *.dat *.vti

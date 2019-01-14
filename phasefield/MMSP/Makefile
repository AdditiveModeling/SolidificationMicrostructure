# Makefile
# GNU makefile for dendritic solidification code
# Questions/comments to scott@dpeters.net (Scott Peters)

# includes
incdir = $(CURDIR)/include

# compilers/flags
compiler = g++ -O3
pcompiler = mpic++ -O3
flags = -I $(incdir)
pflags = $(flags) -include mpi.h

# the program


serial: initialize.cpp simulate.cpp
	$(compiler) $(flags) initialize.cpp -o initialize.out -lz
	$(compiler) $(flags) simulate.cpp -o serial.out -lz

parallel: simulate.cpp
	$(compiler) $(flags) initialize.cpp -o initialize.out -lz
	$(pcompiler) $(pflags) simulate.cpp -o parallel.out -lz

clean:
	rm -rf serial.out initialize.out parallel.out *~ *.dat *.vti

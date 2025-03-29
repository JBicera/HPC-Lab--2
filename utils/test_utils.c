#include <mpi.h>

int getP(){
	int P;
  	MPI_Comm_size(MPI_COMM_WORLD, &P);
  	return P;
}

int getRank(){
  	int rank;
  	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  	return rank;
}

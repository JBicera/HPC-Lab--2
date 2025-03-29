#include <assert.h>
#include <stdlib.h>
#include <mpi.h>

int GT_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{
	assert(datatype == MPI_INT);
	assert(root == 0);
	
	/* Your code here (Do not just call MPI_Bcast) */
	int rank, P;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &P);


	int bitmask = 1; 
	MPI_Request request;

	
	while (bitmask < P) 
	{
		int partner = rank ^ bitmask; // XOR to find partner in current stage
		
		// Check if this process is in the sending part of the tree (rank isn't in bitmask set)
		if (!(rank & bitmask)) 
		{
			// This process sends its data and then exits.
			if (partner < P) {
                MPI_Isend(buffer, count, MPI_INT, partner, 0, comm, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE); // Ensure data is sent
            }
		}
		// If on receiving part (rank is in bitmask set)
		else if (partner < P) 
		{
			// This process receives data from its partner and adds it.
			MPI_Irecv(buffer, count, MPI_INT, partner, 0, comm, &request);
			MPI_Wait(&request, MPI_STATUS_IGNORE); // Ensure data is received
		}

		// Shift bitmask
		bitmask <<= 1;
	}

	return MPI_SUCCESS;
}

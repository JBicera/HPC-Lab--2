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
	
	while (bitmask < P) 
	{
		int partner = rank ^ bitmask; // XOR to find partner in current stage
		
		if (!(rank & bitmask)) 
		{
			if (partner < P) 
			{
				// Send the data the node has to other nodes
                MPI_Send(buffer, count, MPI_INT, partner, 0, comm);
            }
		}
		else if (partner < P) 
		{
			// Receive the data into buffer from partner
            MPI_Recv(buffer, count, MPI_INT, partner, 0, comm, MPI_STATUS_IGNORE); 
		}

		// Shift bitmask
		bitmask <<= 1;
	}

	return MPI_SUCCESS;
}

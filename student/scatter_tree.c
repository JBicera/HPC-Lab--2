#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int GT_Scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype,
               int root, MPI_Comm comm) {
    assert(sendtype == MPI_INT && recvtype == MPI_INT);
    assert(root == 0);

    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);
    MPI_Request request;
    
    
    int *tempBuffer = malloc(P * sendcount * sizeof(int));
    if (rank == root) {
        memcpy(tempBuffer, sendbuf, P * sendcount * sizeof(int));  // Copy the full sendbuf into tempBuffer
        int offset = rank * sendcount;
        memcpy(recvbuf, tempBuffer + offset, recvcount * sizeof(int));
    }


    int bitmask = 1;
    while (bitmask < P) 
    {
        int partner = rank ^ bitmask;  // XOR finds partner

        // Check if this process is in the sending part of the tree (rank isn't in bitmask set)
		if (!(rank & bitmask)) 
		{
			// This process sends its data and then exits.
			if (partner < P) {
                MPI_Isend(tempBuffer, P * sendcount, MPI_INT, partner, 0, comm, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE); // Ensure data is sent
            }
		}
		// If on receiving part (rank is in bitmask set)
		else if (partner < P) 
		{
			// This process receives data from its partner and adds it.
			MPI_Irecv(tempBuffer, P * sendcount, MPI_INT, partner, 0, comm, &request);
			MPI_Wait(&request, MPI_STATUS_IGNORE); // Ensure data is received
            // Calculate the starting offset based on the rank
            int offset = rank * sendcount;

            // Extract the appropriate slice of data
            memcpy(recvbuf, tempBuffer + offset, recvcount * sizeof(int));
        }

        bitmask <<= 1;
    }
    free(tempBuffer);
    return MPI_SUCCESS;
}

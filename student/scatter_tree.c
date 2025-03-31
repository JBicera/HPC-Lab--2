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
    
    // Allocate tempbuffer that can hold all of the data
    int *tempBuffer = malloc(P * sendcount * sizeof(int));

    if (rank == root) 
    {
        memcpy(tempBuffer, sendbuf, P * sendcount * sizeof(int));  // Copy the full sendbuf into tempBuffer
        int offset = rank * sendcount;
        memcpy(recvbuf, tempBuffer + offset, recvcount * sizeof(int)); // Copy needed data into root's own recvbuf (final)
    }


    int bitmask = 1;

    while (bitmask < P) 
    {
        int partner = rank ^ bitmask;  // XOR to find partner in current stage

		if (!(rank & bitmask)) 
		{
			// Node sends all of its data from tempBuffer to next node (Basically a broadcast)
			if (partner < P) 
            {
                MPI_Isend(tempBuffer, P * sendcount, MPI_INT, partner, 0, comm, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE); // Ensure data is sent
            }
		}
		else if (partner < P) 
		{
			// Receives data from its partner 
			MPI_Irecv(tempBuffer, P * sendcount, MPI_INT, partner, 0, comm, &request);
			MPI_Wait(&request, MPI_STATUS_IGNORE);

            // Calculate the starting offset based on the node's rank
            int offset = rank * sendcount;

            // Extract the appropriate slice of data and copy into recvbuf
            memcpy(recvbuf, tempBuffer + offset, recvcount * sizeof(int));
        }

        bitmask <<= 1;
    }
    
    free(tempBuffer);
    return MPI_SUCCESS;
}

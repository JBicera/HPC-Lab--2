#include <assert.h>
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


int GT_Allgather(void *sendbuf, 
                  int sendcount,
                  MPI_Datatype sendtype, 
                  void *recvbuf, 
                  int recvcount,
                  MPI_Datatype recvtype, 
                  MPI_Comm comm)
{
    assert(sendtype == MPI_INT && recvtype == MPI_INT);
    
    /* Your code here (Do not just call MPI_Allgather) */
    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    int *intRecv = (int *)recvbuf;
    int *intSend = (int *)sendbuf;
    
    // Copy its own data into its designated slot
    memcpy(intRecv + rank * sendcount, intSend, sendcount * sizeof(int));

    int prev = rank - 1;
    if (prev < 0) 
        prev = P - 1;
    
    int next = rank + 1;
    if (next == P) 
        next = 0;

    int curi = rank;
    MPI_Request sendRequest, recvRequest;
    MPI_Status status;

    // Send and receive data in a cycle
    for (int i = 0; i < P - 1; i++) 
    {
        // Send the current block  to the next process
        MPI_Isend(intRecv + curi * sendcount, sendcount, MPI_INT, next, 0, comm, &sendRequest);

        // Adjust curi if it overflows
        curi -= 1;
        if(curi < 0)
            curi = P-1;

        // Receive a block from previous process
        MPI_Irecv(intRecv + curi * sendcount, sendcount, MPI_INT, prev, 0, comm, &recvRequest);
        
        // Wait for both to finish before continuing 
        MPI_Wait(&sendRequest, &status);
        MPI_Wait(&recvRequest, &status);
    }


    return MPI_SUCCESS;
}




int GT_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{
    assert(datatype == MPI_INT);
    assert(root == 0);

    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);
    int perProcess = count / P;

    // Allocate a temporary buffer for the scatter phase (per process chunk)
    void *recvbuf = malloc(perProcess * sizeof(int));
    
    // Scatter data from the root process to all processes
    GT_Scatter(buffer, perProcess, datatype, recvbuf, perProcess, datatype, root, comm); // Each process will receive 'perProcess' elements

    // Ensure scatter is complete
    MPI_Barrier(comm);

    // Allocate a buffer to hold the complete gathered data.
    void *gatherBuffer = malloc(count * sizeof(int));

    // Use Allgather so that every process gets the full broadcasted data
    // Each process contributes perProcess elements; the total gathered is count number of elements
    GT_Allgather(recvbuf, perProcess, datatype, gatherBuffer, perProcess, datatype, comm);

    // Copy the final gathered result from gatherBuffer into buffer
    memcpy(buffer, gatherBuffer, count * sizeof(int));

    free(gatherBuffer);
    free(recvbuf);
    return MPI_SUCCESS;
}


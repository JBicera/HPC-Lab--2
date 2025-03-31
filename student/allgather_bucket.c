#include <assert.h>
#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

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

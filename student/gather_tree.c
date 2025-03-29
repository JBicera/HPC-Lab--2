#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int GT_Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
              void *recvbuf, int recvcount, MPI_Datatype recvtype,
              int root, MPI_Comm comm) {
    assert(sendtype == MPI_INT && recvtype == MPI_INT);
    assert(root == 0);  

    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);
    MPI_Status status;
    MPI_Request request;

    // Allocate  temporary buffer to accumulate all data
    int *accumValues = malloc(P * sendcount * sizeof(int));

    // Each process starts by copying its local data into the beginning of its accumulation buffer.
    // (We assume sendcount is 1 in your case, but this code works for any sendcount.)
    memcpy(accumValues, sendbuf, sendcount * sizeof(int));

    // currentCount tracks the number of elements accumulated so far.
    int currentCount = sendcount;  

    int bitmask = 1;

    while (bitmask < P) 
    {
        int partner = rank ^ bitmask;

        if (partner >= P) 
        {
            bitmask <<= 1;
            continue;
        }
        // Determine the amount of data to exchange this round for both senders and receivers
        int blockSize = bitmask * sendcount;  

        if (rank & bitmask)
        {
            // Send entire accumulated buffer (of size currentCount) to its partner.
            MPI_Isend(accumValues, currentCount, MPI_INT, partner, 0, comm, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
            break;
        } 
        else 
        {
            // Receive exactly blockSize elements from its partner and appends them at the currentCount offset in accumValues
            MPI_Irecv(accumValues + currentCount, blockSize, MPI_INT, partner, 0, comm, &request);
            MPI_Wait(&request, &status);
            currentCount += blockSize;
        }

        bitmask <<= 1;
    }

    // Copy accumValues to root's recvbuf
    if (rank == root)
        memcpy(recvbuf, accumValues, P * sendcount * sizeof(int));

    free(accumValues);
    return MPI_SUCCESS;
}

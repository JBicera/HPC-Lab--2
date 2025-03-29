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
    
    /* Your code here (Do not just call MPI_Gather) */
    
    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);
    MPI_Status status;
    MPI_Request request;

    // Allocate buffers
    int *localBuffer = malloc(P * sendcount * sizeof(int));  // Buffer that accumulates all data
    int *recvTemp = malloc(P * sendcount * sizeof(int));     // Temporary buffer for received data

    // Initialize localBuffer with this process's own data
    memcpy(localBuffer, sendbuf, sendcount * sizeof(int));
    int totalElements = sendcount; // Number of elements currently stored in localBuffer

    int bitmask = 1;

    while (bitmask < P) 
    {
        int partner = rank ^ bitmask;  // Compute partner rank

        if (partner >= P) 
        {
            bitmask <<= 1;
            continue;
        }

        if (rank & bitmask) 
        {
            // Sender: Send localBuffer to partner
            MPI_Isend(localBuffer, totalElements, MPI_INT, partner, 0, comm, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
            break;  // Once sent, the process is done
        } 
        else 
        {
            // Receiver: Receive data into recvTemp
            int receivedElements;
            MPI_Irecv(recvTemp, P * sendcount, MPI_INT, partner, 0, comm, &request);
            MPI_Wait(&request, &status);
            MPI_Get_count(&status, MPI_INT, &receivedElements);

            // Append received data into localBuffer
            memcpy(localBuffer + totalElements, recvTemp, receivedElements * sizeof(int));
            totalElements += receivedElements;  // Update total element count
        }

        bitmask <<= 1;
    }

    // Root process copies the final gathered data into recvbuf from localBuffer
    if (rank == root) {
        memcpy(recvbuf, localBuffer, totalElements * sizeof(int));
    }

    free(localBuffer);
    free(recvTemp);
    return MPI_SUCCESS;
}

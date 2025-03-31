#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

int GT_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
			MPI_Op op, int root, MPI_Comm comm)
{
   assert(datatype == MPI_INT);
   assert(op == MPI_SUM); // Asserts summation
   assert(root == 0);
					
   /* Your code here (Do not just call MPI_Reduce) */
	int rank, P;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &P);

   // Initialize  temp buffer for receiving values
   int *tempRecv = (int*)malloc(count * sizeof(int));

   int bitmask = 1; 

   while (bitmask < P) 
   {
      int partner = rank ^ bitmask; // XOR to find partner in current stage of reduction

      if (rank & bitmask) 
      {
         MPI_Send((void*)sendbuf, count, MPI_INT, partner, 0, comm);
         break;  // Once sent, exit loop 
      } 
      else if (partner < P) 
      {  
         // Receive data into temp buffer then append to local values (sendbuf)
         MPI_Recv(tempRecv, count, MPI_INT, partner, 0, comm, MPI_STATUS_IGNORE);

         // Element-wise summation to update nodes sendbuf
         for (int i = 0; i < count; i++)
            ((int*)sendbuf)[i] += tempRecv[i];
      }
      
      // Shift bitmask
      bitmask <<= 1; 
   }

   // After the reduction is complete, copy result to root receive buffer
   if (rank == root) 
   {
      for (int i = 0; i < count; i++)
         ((int*)recvbuf)[i] = ((int*)sendbuf)[i];
   }

   free(tempRecv);

   return MPI_SUCCESS;
}

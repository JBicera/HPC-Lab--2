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

   // Initialize local and temp buffer for reduction
   int *localValue = (int*)malloc(count * sizeof(int));
   int *tempValue = (int*)malloc(count * sizeof(int));

   int bitmask = 1; 
   MPI_Request request;

   // Copy initial values from sendbuf
   for (int i = 0; i < count; i++) 
   {
      localValue[i] = ((int*)sendbuf)[i];
   }

   while (bitmask < P) 
   {
      int partner = rank ^ bitmask; // XOR to find partner in current stage of reduction

      if (rank & bitmask) 
      {  
         // Send data to its partner
         MPI_Isend(localValue, count, MPI_INT, partner, 0, comm, &request);
         MPI_Wait(&request, MPI_STATUS_IGNORE); // Ensure data is sent
         break;  // Once a rank sends its data, break
      } 
      else if (partner < P) // Is receiving process, check if partner is within range
      {  
         // Receiving data then append to local
         MPI_Irecv(tempValue, count, MPI_INT, partner, 0, comm, &request);
         MPI_Wait(&request, MPI_STATUS_IGNORE); // Ensure data is received
         
         // Element-wise summation
         for (int i = 0; i < count; i++)
            localValue[i] += tempValue[i];
      }
      
      // Shift bitmask
      bitmask <<= 1; // Move to the next step in the reduction tree
   }

   // After the reduction is complete, copy result to root receive buffer
   if (rank == root) 
   {
      for (int i = 0; i < count; i++)
         ((int*)recvbuf)[i] = localValue[i];
   }

   // Free Memory
   free(localValue);
   free(tempValue);

   return MPI_SUCCESS;
}

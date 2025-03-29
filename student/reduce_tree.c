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
   int *tempRecv = (int*)malloc(count * sizeof(int));

   int bitmask = 1; 
   MPI_Request request;

   while (bitmask < P) 
   {
      int partner = rank ^ bitmask; // XOR to find partner in current stage of reduction

      if (rank & bitmask) 
      {  
         // Send data to its partner
         MPI_Isend((void*)sendbuf, count, MPI_INT, partner, 0, comm, &request);
         MPI_Wait(&request, MPI_STATUS_IGNORE); // Ensure data is sent
         break;  // Once a rank sends its data, break
      } 
      else if (partner < P) // Is receiving process, check if partner is within range
      {  
         // Receiving data then append to local
         MPI_Irecv(tempRecv, count, MPI_INT, partner, 0, comm, &request);
         MPI_Wait(&request, MPI_STATUS_IGNORE); // Ensure data is received
         
         // Element-wise summation
         for (int i = 0; i < count; i++)
            ((int*)sendbuf)[i] += tempRecv[i];
      }
      
      // Shift bitmask
      bitmask <<= 1; // Move to the next step in the reduction tree
   }

   // After the reduction is complete, copy result to root receive buffer
   if (rank == root) 
   {
      for (int i = 0; i < count; i++)
         ((int*)recvbuf)[i] = ((int*)sendbuf)[i];
   }

   // Free Memory
   free(tempRecv);

   return MPI_SUCCESS;
}

/*
Notes:
Buffer - Address space that references the data that is to be sent/received
   - Usually a variable name is sent/received
   - Arg is pass by reference in C++
   - Must be prepended with ampersand: &var1
Data Count - Number of data elements of a particular type to be sent
Data Type - MPI's predefeined elementary data types
Destination - Arg to send routines that indiciates which process the message should be delivered
   - Specified as the rank of the receiving process
Source - Same as above but for originating process of the message
Tag - Arbitrary non-negtaive integer assigned by programmer to uniquely identify a message
   - Send and receive should have matching message tags
   - For a receive operation can use MPI_ANY_TAG to receive any messages regardless of tag
COMM - Indicates the communication context, or set of proccesses for which the source/dest fields are valid
   - Unless programmer is creating new communciators, the predfined communicator MPI_COMM_WORLD is usually used
Status - MPI_Status object contains information about result of a message operation such as send or receive after it completes
   - Usually used to check source of message, the tag, and size of message sent
   - After successful receive, you can query status object to get details of
      - Source, Tag, Error code
Request - Object used for handling non-blocking communication operations such as MPI_Isend/MPI_Irecv
   - Non-blocking functions start communication that may not complete immediately, allowing the program to continue executing while
   the communication is still in process
   - Need to use MPI_Wait or MPI_test to check if the non-blocking op is finished

MPI_Send(buffer,count,type,dest,tag,comm)
   - Blocking Send Operation
   - Returns only after the application buffer in the sending task is free for reuse
MPI_Isend(buffer,count,type,dest,tag,comm,request)
   - Sends buffer and processing continues immediately without waiting for message to be copied out of applicaiton buffer
   - Program should not modify the applicaiton buffer until MPI_Wait or MPI_Test indicates the send is completed
MPI_Recv(buffer,count,type,source,tag,comm,status)
   - Receive a message and block until the requested data is available in the application buffer in the receiving task
MPI_Irecv(buffer,count,type,soruce,tag,comm,request)
MPI_Wait(&request,&status)
   - Blocks until a specidied non-blocking send or receive operation has completed
   - Has variations like Waitsome, Waitall
MPI_Test(&request, &flag, &status)
   - Checks status of a specified non-blocking send or receive operation
   - Flag parameter is returned true (1) if completed and false (0) if not
*/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <string.h>

#include "test_utils.h"
#include "alltoall_test.h"

// NOTE - THIS ISN'T WORKING YET!

struct alltoall_instance_t{
  void *sendbuf;
  int sendcount;
  MPI_Datatype sendtype;
  void *recvbuf;
  int recvcount;
  MPI_Datatype recvtype;
  MPI_Comm comm;

  void *ref_recvbuf;
};

alltoall_instance_t* alltoall_instance_create(int count){
  alltoall_instance_t* self = (alltoall_instance_t*) malloc(sizeof(alltoall_instance_t));

  self->sendbuf = malloc(count * sizeof(int));
  
  self->sendcount = count;
  self->sendtype = MPI_INT;

  self->recvbuf = malloc(count * sizeof(int) * getP());
  self->ref_recvbuf = malloc(count * sizeof(int) * getP());
  
  self->recvcount = count;
  self->recvtype = MPI_INT;
  self->comm = MPI_COMM_WORLD;

  alltoall_instance_reset(self);

  MPI_Alltoall(self->sendbuf, self->sendcount, self->sendtype, 
      self->ref_recvbuf, self->recvcount, self->recvtype, self->comm);

  return self;
}

void alltoall_instance_destroy(alltoall_instance_t* self){
  free(self->sendbuf);

  free(self->recvbuf);
  free(self->ref_recvbuf);

  free(self);
}

void alltoall_instance_reset(alltoall_instance_t* self){
  int* buf = (int*) self->sendbuf;
  int partNum = getRank();

  for(int i = 0; i < self->sendcount; i++)
    buf[i] = i*(partNum+1);

  memset(self->recvbuf, 0, self->recvcount * sizeof(int) * getP());
}

int alltoall_instance_call(alltoall_instance_t* self, 
                           int (*alltoall)(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                                          void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                                          MPI_Comm comm)){
  return  alltoall(self->sendbuf, self->sendcount, self->sendtype, 
                    self->recvbuf, self->recvcount, self->recvtype,
                    self->comm);
}

int alltoall_instance_fails(alltoall_instance_t* self, 
                           int (*alltoall)(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                                            void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                                            MPI_Comm comm)){
  int local_result, global_result;

  alltoall(self->sendbuf, self->sendcount, self->sendtype, 
            self->recvbuf, self->recvcount, self->recvtype,
            self->comm);

  local_result = (memcmp(self->recvbuf, self->ref_recvbuf, self->recvcount * getP() * sizeof(int)) != 0);

  MPI_Allreduce(&local_result, &global_result, 1, MPI_INT, MPI_LOR, self->comm);

  return global_result;
}

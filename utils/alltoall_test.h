#ifndef _alltoall_H_
#define _alltoall_H_

#include <mpi.h>

#include "test_utils.h"
#include "alltoall_test.h"

typedef struct alltoall_instance_t alltoall_instance_t;

alltoall_instance_t* alltoall_instance_create(int count);

void alltoall_instance_destroy(alltoall_instance_t* self);

void alltoall_instance_reset(alltoall_instance_t* self);

int alltoall_instance_call(alltoall_instance_t* self, 
                           int (*alltoall)(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                                          void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                                          MPI_Comm comm));

int alltoall_instance_fails(alltoall_instance_t* self, 
                           int (*alltoall)(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                                            void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                                            MPI_Comm comm));

#endif
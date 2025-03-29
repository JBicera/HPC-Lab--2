/**
 *  \file driver.c
 *
 *  \brief Driver file for CSE 6230, Fall 2016, Lab 3:
 * 
 *
 *  \author Rich Vuduc <richie@gatech...>
 *
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include <mpi.h>
#include <getopt.h>

#if defined(DRIVEGATHER)

#include "gather_test.h"

extern int GT_Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                     void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                     int root, MPI_Comm comm);

#define CREATEINSTANCE gather_instance_t* instance = gather_instance_create(count)

#define IMPLFAILS gather_instance_fails(instance, GT_Gather)

#define IMPLCALL gather_instance_call(instance, GT_Gather)

#define DESTROYINSTANCE gather_instance_destroy(instance)

#elif defined(DRIVESCATTER)

#include "scatter_test.h"

extern int GT_Scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                     void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                     int root, MPI_Comm comm);

#define CREATEINSTANCE scatter_instance_t* instance = scatter_instance_create(count)

#define IMPLFAILS scatter_instance_fails(instance, GT_Scatter)

#define IMPLCALL scatter_instance_call(instance, GT_Scatter)

#define DESTROYINSTANCE scatter_instance_destroy(instance)

#elif defined(DRIVEREDUCE)

#include "reduce_test.h"

extern int GT_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                     MPI_Op op, int root, MPI_Comm comm);

#define CREATEINSTANCE reduce_instance_t* instance = reduce_instance_create(count)

#define IMPLFAILS reduce_instance_fails(instance, GT_Reduce)

#define IMPLCALL reduce_instance_call(instance, GT_Reduce)

#define DESTROYINSTANCE reduce_instance_destroy(instance)

#elif defined(DRIVEBROADCAST)

#include "broadcast_test.h"

extern int GT_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);

#define CREATEINSTANCE broadcast_instance_t* instance = broadcast_instance_create(count)

#define IMPLFAILS broadcast_instance_fails(instance, GT_Bcast)

#define IMPLCALL broadcast_instance_call(instance, GT_Bcast)

#define DESTROYINSTANCE broadcast_instance_destroy(instance)

#elif defined(DRIVEALLGATHER)

#include "allgather_test.h"

extern int GT_Allgather(void *sendbuf, int  sendcount, MPI_Datatype sendtype, 
                        void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                        MPI_Comm comm);

#define CREATEINSTANCE allgather_instance_t* instance = allgather_instance_create(count)

#define IMPLFAILS allgather_instance_fails(instance, GT_Allgather)

#define IMPLCALL allgather_instance_call(instance, GT_Allgather)

#define DESTROYINSTANCE allgather_instance_destroy(instance)

#elif defined(DRIVEALLTOALL)

#include "alltoall_test.h"

extern int GT_Alltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                     void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                     MPI_Comm comm);

#define CREATEINSTANCE alltoall_instance_t* instance = alltoall_instance_create(count)

#define IMPLFAILS alltoall_instance_fails(instance, GT_Alltoall)

#define IMPLCALL alltoall_instance_call(instance, GT_Alltoall)

#define DESTROYINSTANCE alltoall_instance_destroy(instance)

#endif

/** Smallest message to test (in words) */
#define MIN_COUNT (1 << 0)

/** Largest message to test (in words) */
#define MAX_COUNT (1 << 26)

/** Minimum time (seconds) for a set of trials */
#define MIN_TIME 0.2

/* Skip width - skip factors of two because its unnecessary */
/* A SKIP_WIDTH of 1 will try every factor of two size. */
#define SKIP_WIDTH 4

static struct option long_options[] =
{
  {"minshift",  required_argument, 0, 'l'},
  {"shift",  required_argument, 0, 'e'},
  {"maxshift",  required_argument, 0, 'u'},
  {"help",  no_argument, 0, 'h'},
  {0, 0, 0, 0}
};

#define USAGE                                                                 \
"usage:\n"                                                                    \
"  driver [options]\n"                                                     \
"options:\n"                                                                  \
"  -l [minshift]    Sets the minimum number of elements to  1 << minshift\n"\
"  -e [shift]       Sets the minimum and maximum number of elements to 1 << shift\n"\
"  -u [minshift]    Sets the maximum number of elements to 1 << minshift\n"\
"  -h [help]        Prints this help message"

/* ============================================================ */

int main (int argc, char *argv[])
{
  int rank, P;
  MPI_Init (&argc, &argv);

  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &P);

  unsigned long min_count = MIN_COUNT;
  unsigned long max_count = MAX_COUNT;

  int c, option_index;

  while((c = getopt_long (argc, argv, "l:e:u:h", long_options, &option_index)) >= 0){
    switch(c){
      case 'l':
        min_count = 1 << atoi(optarg);
        break;
      case 'e':
        min_count = 1 << atoi(optarg);
        max_count = min_count;
        break;
      case 'u':
        max_count = 1 << atoi(optarg);
        break;
      case 'h':
        fprintf(stderr, USAGE);
        exit(EXIT_SUCCESS);
      case '?':
          /* getopt_long already printed an error message. */
          break;
      default:
        abort ();
    }
  }

  if(min_count > max_count){
    fprintf(stderr, "Your options set the minimum message length to be creater than the maximum.\n");
    exit(EXIT_FAILURE);
  }

  if (rank == 0) fprintf(stdout, "#P\tBytes\tSeconds\tTrials\n");

  for (unsigned long count = min_count; count <= max_count; count <<= SKIP_WIDTH)
  {

    CREATEINSTANCE;

    if (IMPLFAILS){
      if (rank == 0)
        fprintf(stderr, "Your implementation returned an incorrect result for counts of length %lu.\n", count);
      exit(EXIT_FAILURE);
    }

    int trials = 3;
    double t_max = 0;
    do {
      MPI_Barrier (MPI_COMM_WORLD);
      double t_start = MPI_Wtime ();
      for (int k = 0; k < trials; ++k) IMPLCALL;
      double t_elapsed = MPI_Wtime () - t_start;
      MPI_Barrier (MPI_COMM_WORLD);
      MPI_Allreduce (&t_elapsed, &t_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (t_max < MIN_TIME) trials <<= 1;
    } while (t_max < MIN_TIME);

    /* Write out the timing result */
    if (rank == 0)
    {
#ifdef DRIVEBROADCAST
      const unsigned long bytes = P * count * sizeof (int);
#else
      const unsigned long bytes = count * sizeof (int);
#endif
      const double t_collective = t_max / trials;
      fprintf (stdout, "%d\t%lu\t%E\t%d\n", P, bytes, t_collective, trials);
    }

    DESTROYINSTANCE;
  }


  MPI_Finalize();

  return 0;
}

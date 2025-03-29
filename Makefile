MPICC = mpicc
MPICFLAGS = -std=c99
MPICOPTFLAGS = -O0 -g -Wall -Werror
MPILDFLAGS =

allgather_bucket: utils/driver.c student/allgather_bucket.c utils/allgather_test.c utils/test_utils.c
	$(MPICC) $(MPICFLAGS) $(MPICOPTFLAGS) -DDRIVEALLGATHER -o $@ $^

reduce_tree: utils/driver.c student/reduce_tree.c utils/reduce_test.c utils/test_utils.c
	$(MPICC) $(MPICFLAGS) $(MPICOPTFLAGS) -DDRIVEREDUCE -o $@ $^

# Future Addition
#alltoall_any: utils/driver.c student/alltoall_any.c utils/alltoall_test.c utils/test_utils.c
#	$(MPICC) $(MPICFLAGS) $(MPICOPTFLAGS) -DDRIVEALLTOALL -o $@ $^

broadcast_sag: utils/driver.c student/broadcast_sag.c utils/broadcast_test.c utils/test_utils.c
	$(MPICC) $(MPICFLAGS) $(MPICOPTFLAGS) -DDRIVEBROADCAST -o $@ $^

broadcast_tree: utils/driver.c student/broadcast_tree.c utils/broadcast_test.c utils/test_utils.c
	$(MPICC) $(MPICFLAGS) $(MPICOPTFLAGS) -DDRIVEBROADCAST -o $@ $^

gather_tree: utils/driver.c student/gather_tree.c utils/gather_test.c utils/test_utils.c
	$(MPICC) $(MPICFLAGS) $(MPICOPTFLAGS) -DDRIVEGATHER -o $@ $^

scatter_tree: utils/driver.c student/scatter_tree.c utils/scatter_test.c utils/test_utils.c
	$(MPICC) $(MPICFLAGS) $(MPICOPTFLAGS) -DDRIVESCATTER -o $@ $^

driver$(EXEXEXT) : utils/driver.c $(COLLECTIVE_SRC) utils/test_utils.c
	$(MPICC) $(MPICFLAGS) $(MPICOPTFLAGS) -D$(DRIVERMACRO) -o $@ $^ $(MPILDFLAGS)


all: allgather_bucket reduce_tree broadcast_sag broadcast_tree gather_tree scatter_tree

clean:
	rm -f *.o allgather_bucket reduce_tree broadcast_sag broadcast_tree gather_tree scatter_tree

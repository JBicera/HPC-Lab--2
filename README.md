# Lab 2: Collectives -- OpenMPI

  - [Getting Started](#getting-started)
  - [Programming](#programming)
  - [Testing Your Code and Measuring Performance](#testing-your-code-and-measuring-performance)
  - [Submission](#submission)
  - [Grading](#grading)

In this lab, you will implement several key collective operations discussed in the lessons. This gives you an opportunity to solidify your understanding of the relevant algorithms and to see how latency and bandwidth considerations impact overall performance. You will also be familiarizing yourself with the MPI (Message Passing Interface) standard, which you will use in subsequent assignments. MPI provides a portable interface for parallel tasks running on different nodes of a cluster or supercomputer to send messages to each other.

## Getting Started

Begin by obtaining the starter code from the github repository.

```
    git clone https://github.gatech.edu/omscse6220/lab2sp25.git
```

Note that this is the [GT github server](https://github.gatech.edu), so you will need to authenticate with your GT credentials.

Optionally, you may choose use a git hosting service for your code.  As always, please **do not make your repository public** on Github or another git hosting service.  Anyone will be able to see it.  If you feel the need to use a hosting service for your project, please keep your repository private.  Your GT account allows you to create free private repositories on [the GT github server](https://github.gatech.edu).

In past semesters, students have found [this resource on MPI collectives](https://www.cs.utexas.edu/users/flame/pubs/InterCol_TR.pdf) and asked if they can use it as a reference. Rather than have some students miss that note on the forums, we have included it here. **If you use this as a resource, or any other resource, please cite it.** A simple one or two sentence comment with a link to the source is sufficient. Remember, pseudocode is okay. Actual code is cheating.

For gather tree, we are often asked if students can write their code as a depth = 2 tree. We will eventually phase this out, but for the time being know that students have had success setting depth to 2 or allowing for any depth. Either approach is currently acceptable.

## Programming

Your task is to implement the following collectives:

*  Reduce
*  Scatter
*  Gather
*  Allgather
*  Broadcast : Tree-Based Implementation
*  Broadcast : Scatter-Allgather Implementation

The **student** directory contains stubs for each of the collectives.

In the given files, you are to implement an interface with function of the form GT\_\<collective\>, which mirrors the MPI library.  Thus, GT_Bcast should do the same thing as MPI_Bcast.  In your implementation, **you may only use point-to-point MPI operations.**  These are  `MPI_Send`, `MPI_Recv`, `MPI_Isend`, `MPI_Irecv`, `MPI_Wait`, `MPI_Sendrecv` and `MPI_Waitall`.  Simply calling `MPI_Bcast` within your `GT_Bcast` implementation, for instance, is **not** allowed. And because someone will ask, you are of course allowed to use other MPI calls that are *not* communication-related (MPI_Comm_rank, MPI_Comm_size, etc.)

For Broadcast_sag, the Scatter-Allgather implementation, it is suggested to literally copy *your* code for scatter and allgather (at least as a starting point.)

Several good sources of documentation for MPI are available online.  There are several different versions and implementations; however, these differences will not be significant for our purposes.

*  [OpenMPI](http://www.open-mpi.org/doc/)
*  [MPI Forum](http://www.mpi-forum.org/docs/)
*  [LLNL](https://hpc-tutorials.llnl.gov/mpi/)
*  [MPI Tutorial](http://mpitutorial.com/)
*  [Slides from Robert van de Geijn](documentation/collective_communication.pdf)

MPI is a very general tool, and to fully implement all of its features is beyond the scope of this lab.  Therefore, you are permitted to make simplifying assumptions such as:

*  the input buffers are valid memory addresses, not MPI_IN_PLACE for example.
*  the root is zero
*  the datatype is MPI_INT
*  Reduce is only doing summation (MPI_SUM)

In most cases, these assumptions are documented through the `assert` statements in the starter code.

## Testing Your Code and Measuring Performance

In addition to the starter code for the MPI-like functions that you are to implement, the repository contains:

*  A grading script (grader.sh)
*  A Makefile. **Do not make any changes to it.**
*  A performace test program, speedtest.py. You can run this as:
```
python3 speedtest.py
```
Note that speedtest.py surpresses error messages (to get clean output). Don't run speedtest until your code is correct!
*  Test files of the form \<collective\>\_test.c and \<collective\>\_test.h in **utils**, which run some simple tests
*  pace_env.sh, the environment script to set library versions that will be used during grading.
*  example.sub, an example script for submitting performance test jobs to PACE.
*  pace_interactive.sh, an example script to get an interactive shell on PACE.


An example of how to test your code locally would be:

```
make scatter_tree
mpirun -np 3 ./scatter_tree
```

An example of how to test your code on the cluster would be:

```
sbatch example.sub
```

You'll notice that the *example.sub* script defaults to running the **speedtest**. You can edit the file to run various collective tests for correctness. 

Results from this script are placed into the **results** subdirectory in a tab-separated file with a name in the form \<speedtest.tsv\>.tsv.

## Submission

Lab submissions for this class will be through Canvas under the appropriate assignment.

The file you need to submit for this lab is:

```sh
   reduce_tree.c
   scatter_tree.c
   gather_tree.c
   allgather_bucket.c
   broadcast_tree.c
   broadcast_sag.c
```
These files should exist in the **student** directory of your lab folder.

Do **not** submit or modify any other files. **Please include each file at the root of a single level zip file named lab2.zip.**

If you include any additional headers or libraries, please make sure they compile as-is on the VM and PACE.

## Grading

Your grade on this lab will consist of the following:

* Provided test cases
* Additional private cases
* Performance score

The point totals of each are listed in the Ed announcement post.

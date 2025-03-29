#!/bin/sh
salloc -Jlab2_mpi_interactive --partition=ice-cpu,coc-cpu -N4 --ntasks-per-node=2 -C"intel&core24" --mem-per-cpu=16G --time=00:15:00
exit 0

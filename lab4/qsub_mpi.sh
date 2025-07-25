#!/bin/sh
#PBS -N qsub_mpi_ann
#PBS -e test.e
#PBS -o test.o
#PBS -l nodes=4:ppn=8

NODES=$(cat $PBS_NODEFILE | sort | uniq)

for node in $NODES; do
    scp master_ubss1:/home/${USER}/ann/main ${node}:/home/${USER} 1>&2
    scp -r master_ubss1:/home/${USER}/ann/files ${node}:/home/${USER}/ 1>&2
done

/usr/local/bin/mpiexec -np 32 -machinefile $PBS_NODEFILE /home/${USER}/main

scp -r /home/${USER}/files/ master_ubss1:/home/${USER}/ann/ 2>&1

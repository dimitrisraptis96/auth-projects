#!/bin/bash
#PBS -q pdlab
#PBS -N serial2
#PBS -j oe
#PBS -l nodes=1:ppn=2,walltime=59:00

cd $PBS_O_WORKDIR/

./bin/main 1 60000 30 30
./bin/main 1 60000 30 30
./bin/main 1 60000 30 30
./bin/main 1 60000 30 30

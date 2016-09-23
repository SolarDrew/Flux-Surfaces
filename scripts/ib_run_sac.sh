#!/bin/bash
#$ -l h_rt=99:00:00
#$ -cwd
#$ -l arch=intel*
#$ -l mem=6G
#$ -pe openmpi-ib 16
#$ -N jet-formation
#$ -P mhd
#$ -q mhd.q
#$ -j y

#Set the Python virtualenv
source ~/.bashrc
workon vtk_hdf 

#Load MPI modules
module add mpi/pgi/openmpi/1.10.0

echo "SAC will run on the following nodes"
cat $PE_HOSTFILE

echo "Run SAC:"
time python /home/sm1ajl/Flux-Surfaces/run.py SAC --mpi

echo "Run GDF Translator:"
time python /home/sm1ajl/Flux-Surfaces/run.py gdf --mpi

echo "Job Complete"

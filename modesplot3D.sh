#!/bin/bash -e
#$ -l h_rt=70:00:00
#$ -cwd
#$ -l arch=intel*
# #$ -l mem=8G
#$ -P mhd
#$ -q mhd.q
#$ -N threeDplots
# #$ -t 2-10
#$ -t 2
#$ -o $JOB_NAME.o$JOB_ID
#$ -e $JOB_NAME.o$JOB_ID

echo
echo $(date)
echo

source $HOME/.bashrc
module purge
module load apps/python/conda
#module load mpi/gcc/openmpi/1.10.0
#source activate mpi-sac
source activate yt
module load compilers/gcc/6.2
echo $(which python)
echo $(gcc --version)
echo $(glxinfo)

#### Run plots ####
#xvfb-run --auto-servernum mpirun -np 4 python ./modesplot3D.py
#mpirun -np 4 python ./modesplot3D.py

t=$((SGE_TASK_ID))
filename='/fastdata/sm1ajl/Flux-Surfaces/gdf/m0_p120-0_0-5_0-5/m0_p120-0_0-5_0-5_00'$(printf %03d $t)'.gdf'
xvfb-run -a python modesplot3D.py $t $filename
#python modesplot3D.py $t $filename

#!/bin/bash -e
#$ -l h_rt=30:00:00
#$ -cwd
#$ -l arch=intel*
#$ -l mem=16G
#$ -l rmem=16G
#$ -pe openmpi-ib 1
#$ -P mhd
#$ -q mhd.q
#$ -N m0_analysis
#$ -t 1-3

source $HOME/.bashrc
module purge
module load apps/python/conda
module load mpi/gcc/openmpi/1.10.0
#module load mpi/intel/openmpi/1.10.0
source activate mpi-sac
echo $(which python)
echo $(which mpirun)
echo $(which mpif90)
#################################################################
################ Set the Parameters for the array ###############
#################################################################


#################################################################
####################### Run the Script ##########################
#################################################################

#### Setup and Configure ####
unset LD_PRELOAD
i=$((SGE_TASK_ID - 1))

BASE_DIR=/data/sm1ajl/Flux-Surfaces/
TMP_DIR=$(mktemp -d --tmpdir=/fastdata/sm1ajl/temp_run/)

cp -r $BASE_DIR $TMP_DIR
cd $TMP_DIR/Flux-Surfaces/
pwd

#### Run the CODE! ####
tube_radii=( 'r60' 'r30' 'r10' )
tuber=${tube_radii[i]}
echo $tuber
#xvfb-run --auto-servernum python ./run.py analysis --mpi --np=8 --tube-r=$tuber
xvfb-run --auto-servernum python ./run.py analysis --mpi --np=1 --tube-r=$tuber
#xvfb-run --auto-servernum python analysis/surface_analysis_mpi.py --tube-r=$tuber
#set +e
#kill $XVFBPID
#set -e

###### I Done it now ########
rm -r $TMP_DIR

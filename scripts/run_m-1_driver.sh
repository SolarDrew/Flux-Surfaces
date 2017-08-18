#!/bin/bash -e
#$ -l h_rt=70:00:00
#$ -cwd
# #$ -l arch=intel*
#$ -l rmem=8G
#$ -pe openmpi-ib 32
# #$ -P mhd
# #$ -q mhd.q
#$ -N driver_m-1
# #$ -j y
#$ -o $JOB_NAME.o$JOB_ID
#$ -e $JOB_NAME.o$JOB_ID
# #$ -t 2-5

echo
echo $(date)
echo

source $HOME/.bashrc
module purge
module load apps/python/conda
#module load mpi/gcc/openmpi/1.10.0
#source activate mpi-sac
module load mpi/openmpi/2.0.1/gcc-6.2
source activate mpi-mayavi2
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
i=$((SGE_TASK_ID - 1))

BASE_DIR=/data/sm1ajl/Flux-Surfaces/
TMP_DIR=$(mktemp -d --tmpdir=/fastdata/sm1ajl/temp_run/)

cp -r $BASE_DIR $TMP_DIR
cd $TMP_DIR/Flux-Surfaces/
pwd

cp scripts/vac_config.par.modedrivers scripts/vac_config.par
#./configure.py set SAC --usr_script="m1" --grid_size=132,260,260 --runtime=200 --mpi_config=np010408
./configure.py set SAC --usr_script="m-1" --grid_size=132,132,132 --runtime=200 --mpi_config=np010408
./configure.py set data --ini_dir="/fastdata/sm1ajl/inidata/" --ini_filename="fredatmos"
./configure.py set driver --exp_fac=0.0 --period=60 --amp="20.d0" --fort_amp="20.d0"
./configure.py print
./configure.py compile sac --clean

#### Run SAC ####
# python ./run.py SAC --mpi
python ./run.py gdf --mpi

#### Run the CODE! ####
# tube_radii=( 'r63' 'r60' 'r57' 'r54' 'r51' 'r48' 'r45' 'r42' 'r39' 'r36' 'r33' 'r30' 'r27' 'r24' 'r21' 'r18' 'r15' 'r12' 'r09' 'r06' 'r03' )
# for tuber in "${tube_radii[@]}"
# do
#     echo $tuber
#     xvfb-run --auto-servernum python ./run.py analysis --mpi --np=32 --tube-r=$tuber
#     set +e
#     kill $XVFBPID
#     set -e
# done

###### I Done it now ########
rm -r $TMP_DIR

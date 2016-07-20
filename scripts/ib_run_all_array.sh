#!/bin/bash -e
#$ -l h_rt=15:00:00
#$ -cwd
#$ -l arch=intel*
#$ -l mem=6G
#$ -pe openmpi-ib 16
#$ -P mhd
#$ -q mhd.q
#$ -N all_periods
#$ -i y
#$ -t 1

source $HOME/.bashrc
module purge
module load apps/python/conda
module load /mpi/gcc/openmpi/1.10.0
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
#i=$((SGE_TASK_ID - 1))
i=0

BASE_DIR=/data/sm1ajl/Flux-Surfaces/
TMP_DIR=$(mktemp -d --tmpdir=/fastdata/sm1ajl/temp_run/)

cp -r $BASE_DIR $TMP_DIR
cd $TMP_DIR/Flux-Surfaces/
pwd

./configure.py set SAC --usr_script="Slog"
./configure.py set driver --exp_fac=0.15 --period=180

#### Run SAC ####
deltas=( "0.15" "0.20" "0.25" "0.30" "0.35" )
amps=( "7.2274441701395986" "5.2788458200228927" "4.167609775756989" "3.4471974045012397" "2.9411894675997523" )

echo ${deltas[i]} ${amps[i]} ${amps[i]:0:3}
./configure.py set driver --delta_x=${deltas[i]} --delta_y=${deltas[i]} --amp=${amps[i]:0:3} --fort_amp=${amps[i]}d0
./configure.py set data --ini_filename=3D_tube_128_128_128
./configure.py print;
./configure.py compile sac --clean;

python ./run.py SAC --mpi
python ./run.py gdf --mpi

#### Run the CODE! ####
tube_radii=( 'r60' 'r30' 'r10' )
for tuber in "${tube_radii[@]}"
do
    echo $tuber
    python ./run.py analysis --mpi --tube-r=$tuber
done

###### I Done it now ########
rm -rf $TMP_DIR

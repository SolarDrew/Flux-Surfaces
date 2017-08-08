#!/bin/bash -e
#$ -l h_rt=24:00:00
#$ -cwd
#$ -l arch=intel*
#$ -l mem=24G
#$ -l rmem=16G
#$ -pe openmpi-ib 2
#$ -P mhd
#$ -q mhd.q
#$ -N radial_displacement
#$ -j y
#$ -t 3-64:3

source $HOME/.bashrc
module purge
module load apps/python/conda
module load mpi/gcc/openmpi/1.10.0
#source activate mpi-sac
source activate yt
echo $(which python)
#echo $(which mpirun)

echo $(date)
#################################################################
################ Set the Parameters for the array ###############
#################################################################


#################################################################
####################### Run the Script ##########################
#################################################################

BASE_DIR=/data/sm1ajl/Flux-Surfaces/
TMP_DIR=$(mktemp -d --tmpdir=/fastdata/sm1ajl/temp_run/)

#cp -r $BASE_DIR $TMP_DIR
rsync -r --exclude=*.out --exclude=*.png --exclude=*.ini --exclude=*.gdf --exclude=*.o*.* --exclude=*.e*.* --exclude=*.po*.* . $TMP_DIR/Flux-Surfaces
cd $TMP_DIR/Flux-Surfaces/
pwd

#### Run SAC ####
# modes=( 'm0' 'm1' 'm2' )
i=$((SGE_TASK_ID))

cp scripts/vac_config.par.modedrivers scripts/vac_config.par
#./configure.py set SAC --usr_script="$m" --grid_size=132,132,132 --runtime=200 --mpi_config=np010408
./configure.py set SAC --grid_size=132,132,132 --runtime=200 --mpi_config=np010408
./configure.py set data --ini_filename="fredatmos"
./configure.py set driver --exp_fac=0.0 --period=60 --amp="20.d0" --fort_amp="20.d0"
./configure.py print

cd analysis

#### Run the CODE! ####
xvfb-run --auto-servernum python radial_distance_rings.py $i $1
set +e
kill $XVFBPID
set -e

###### I Done it now ########
rm -rf $TMP_DIR

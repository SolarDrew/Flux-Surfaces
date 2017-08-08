#!/bin/bash -e
#$ -l h_rt=1:00:00
#$ -cwd
# #$ -l arch=intel*
#$ -pe openmpi-ib 8
#$ -l mem=10G
#$ -l rmem=8G
# #$ -P mhd
# #$ -q mhd.q
#$ -N driver_mx_analysis
#$ -t 1-21
#$ -o $JOB_NAME.o$JOB_ID.$TASK_ID
#$ -e $JOB_NAME.o$JOB_ID.$TASK_ID

echo
echo $(date)
echo

source $HOME/.bashrc
module purge
module load apps/python/conda
module load mpi/gcc/openmpi/1.10.0
#source activate mpi-sac
source activate mayavi2
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
#TMP_DIR=$(mktemp -d --tmpdir=/fastdata/sm1ajl/temp_run/)
TMP_DIR=$(mktemp -d -p /fastdata/sm1ajl/temp_run/)

# cp -r $BASE_DIR $TMP_DIR
rsync -rl --exclude=*.out --exclude=*.png --exclude=*.ini --exclude=*.gdf . $TMP_DIR/Flux-Surfaces
cd $TMP_DIR/Flux-Surfaces/
pwd

#modes=( 'm0' 'm1' 'm2' )
#modes=( 'm-1' )
modes=( 'm0' )
# modes=( 'm-1' 'm0' 'm1' )

cp scripts/vac_config.par.modedrivers scripts/vac_config.par
#./configure.py set SAC --usr_script="${modes[i]}" --grid_size=132,260,260 --runtime=200 --mpi_config=np010408
./configure.py set SAC --grid_size=132,132,132 --runtime=200 --mpi_config=np010408
./configure.py set data --ini_filename="fredatmos"
./configure.py set driver --exp_fac=0.0 --period=60 --amp="20.d0" --fort_amp="20.d0"
#./configure.py print
#./configure.py compile sac --clean

###  Run the CODE! ####
#tube_radii=( 'r126' 'r120' 'r114' 'r108' 'r102' 'r96' 'r90' 'r84' 'r78' 'r72' 'r66' 'r60' 'r54' 'r48' 'r42' 'r36' 'r30' 'r24' 'r18' 'r12' 'r06')
tube_radii=( 'r63' 'r60' 'r57' 'r54' 'r51' 'r48' 'r45' 'r42' 'r39' 'r36' 'r33' 'r30' 'r27' 'r24' 'r21' 'r18' 'r15' 'r12' 'r09' 'r06' 'r03' )
tuber="${tube_radii[i]}"
#tube_radii=( 'r66' 'r60' 'r54' 'r48' 'r42' 'r36' 'r30' 'r24' 'r18' 'r12' 'r06' )
#tube_radii=( 'r126' 'r90' 'r60' 'r30' 'r06' )
#for tuber in "${tube_radii[@]}"
for mode in "${modes[@]}"
do
    #echo $tuber
    echo $mode, $tuber
    ./configure.py set SAC --usr_script=$mode
    ./configure.py print
    set +e
    xvfb-run --auto-servernum python ./run.py analysis --mpi --np=8 --tube-r=$tuber
    set +e
    kill $XVFBPID
    set -e
done

###### I Done it now ########
rm -r $TMP_DIR

#! /bin/bash
#$ -cwd
#$ -l arch=intel*
#$ -l mem=4G
# #$ -l rmem=4G
#$ -pe openmpi-ib 32
#$ -P mhd
# #$ -q mhd.q
#! -N atmostest
#$ -o $JOB_NAME.out
#$ -e $JOB_NAME.err

module load mpi/gcc/openmpi/1.10.0
#set -e
/bin/bash -e
echo $(which python)
echo $(which mpirun)
echo $(which mpif90)

# Get to correct directory and configure VAC
cd ~/Flux-Surfaces/

# Run initial conditions
#source activate yt
#python scripts/fredatmos.py
#cp /fastdata/sm1ajl/inidata/fredatmos.ini /fastdata/sm1ajl/inidata/fredatmos_np010101_000.ini
#source deactivate

# Set up SAC configuration
source activate mpi-sac
cp scripts/vac_config.par.atmostest scripts/vac_config.par
#./configure.py set SAC --usr_script="test" --grid_size=128,128,128 --mpi_config=np020204
./configure.py set SAC --usr_script="test" --grid_size=256,256,256 --mpi_config=np040402
./configure.py set data --ini_dir="/fastdata/sm1ajl/inidata/" --ini_filename="fredatmos"
./configure.py print
./configure.py compile SAC --clean

# Run SAC as currently configured
./run.py SAC --mpi --np=32

# Convert SAC output to gdf
./run.py gdf --mpi --np=32

echo 'All complete'

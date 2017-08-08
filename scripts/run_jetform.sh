#! /bin/bash
#$ -cwd
#$ -l arch=intel*
#$ -l mem=8G
#$ -l rmem=8G
#$ -pe openmpi-ib 16
#$ -P mhd
#$ -q mhd.q
#! -N jet-formation
module load mpi/gcc/openmpi/1.10.0
#module purge
#set -e
/bin/bash -e
source activate mpi-sac

# Get to correct directory and configure VAC
cd ~/Flux-Surfaces/

# Set up SAC configuration
./configure.py set SAC --usr_script="jet" --grid_size=72,72,41
./configure.py set data --ini_dir="/fastdata/sm1ajl/inidata/"
#./configure.py set data --ini_filename="jet-formation"
./configure.py set driver --exp_fac=0 --period=180 --amp="0.0" --fort_amp="0.d0"
./configure.py print
./configure.py compile SAC --clean

cd sac/sac/src/
./setvac -s
cd ~/Flux-Surfaces/

# Run initial conditions
python scripts/jets33.py

# Run SAC as currently configured
./run.py SAC --mpi --np=16

# Convert SAC output to gdf
./run.py gdf --mpi --np=16

echo 'All complete'

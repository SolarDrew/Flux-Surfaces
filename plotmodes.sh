#!/bin/bash -e
#$ -cwd
# #$ -l arch=intel*
# #$ -P mhd
# #$ -q mhd.q
#$ -j y
#$ -t 150
# #$ -t 1-200:4
# #$ -t 1-591:4
#$ -N plots

source $HOME/.bashrc
module purge
module load apps/python/conda
#source activate yt
source activate mayavi
echo $(which python)
#################################################################
################ Set the Parameters for the array ###############
#################################################################


#################################################################
####################### Run the Script ##########################
#################################################################

i=$((SGE_TASK_ID-1))

#xvfb-run -a python modes2.py $1 $i
xvfb-run -a python modesplot3D_poster2.py $1 $i

#cp figs/* /fastdata/sm1ajl/Flux-Surfaces/figs/ -r

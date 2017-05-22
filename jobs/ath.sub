#!/bin/bash 
#---Number of core
#BSUB -n 40
#BSUB -R "span[ptile=40]"

#---Job's name in LSF system
#BSUB -J cv

#---Error file
#BSUB -eo err_cv

#---Output file
#BSUB -oo out_cv

#---LSF Queue name
#BSUB -q PQ_nbc

##########################################################
# Set up environmental variables.
##########################################################
export NPROCS=`echo $LSB_HOSTS | wc -w`
export OMP_NUM_THREADS=$NPROCS

. $MODULESHOME/../global/profile.modules
source /home/data/nbc/athena/athena/bash_environment


##########################################################
##########################################################

python /home/data/nbc/athena/athena/run_cv.py

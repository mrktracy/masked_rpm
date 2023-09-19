#!/bin/bash -l

#$ -P noc-lab       # Specify the SCC project name you want to use
#$ -l buyin
#$ -l h_rt=12:00:00   # Specify the hard time limit for the job
#$ -N rpm_fullrun     # Give job a name
#$ -m beas			# email when job begins, ends, aborts, or is suspended
#$ -l gpus=4		# specify four GPUs

# add cores and memory per core
# get environment from argo, load to SCC
# can check how to get necessary dependencies based on code
# can look into running multiple threads on same GPU

module load python3/3.8.10
python main_tr.py
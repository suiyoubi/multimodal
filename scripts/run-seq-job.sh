#!/bin/bash
jobname=joc-flava
exp_name=$1
num_nodes=$2
gpus_per_node=$3
num_restarts=$4

# Run this script with:  ./run-seq-job.sh <EXP_NAME> <SLURM_JOB_NUM_NODES> <SLURM_GPUS_PER_NODE> <NUM_RESTARTS>
jobid=$(sbatch -N $num_nodes --gpus-per-node $gpus_per_node --job-name=$jobname job.sub $exp_name 0)
echo "Job $jobid submitted"
for i in {0..$num_restarts}; do
	jobid=$(sbatch --dependency=afternotok:${jobid##* } -N $num_nodes --gpus-per-node $gpus_per_node --job-name=$jobname job.sub $exp_name ${jobid##* })
	echo "Job $jobid Submitted"
done
#!/bin/bash
#SBATCH -A ent_joc_model_multimodal_pyt
#SBATCH -p backfill_dgx2h_m2 # batch, batch_16GB, Backfill... Check draco onboarding guide.
#SBATCH -N 1                  # number of nodes
#SBATCH -t 0:30:00              # wall time
#SBATCH -J "ml-model.flava"     # job name (<< CHANGE ! >>)
#SBATCH --exclusive             # exclusive node access
#SBATCH --mem=0                 # all mem avail
#SBATCH --mail-type=FAIL        # only send email on failure
#SBATCH --ntasks-per-node=16    # n tasks per machine
#SBATCH --overcommit
#SBATCH --gpus-per-node=16     # n gpus per machine <required>
#SBATCH --gres=gpu:16            # Specify the number of GPUs even if exclusiveNeeded for pytorch
#SBATCH --nv-meta="ml-model:flava"


# Run this script with: sbatch -N <SLURM_JOB_NUM_NODES> --gpus-per-node <SLURM_GPUS_PER_NODE> run-draco-multi.sh <EXP_NAME>
# Used for running single job on SLURM

set -euxo pipefail

EXP_NAME=$1
HOME_DIR=/gpfs/fs1/projects/ent_joc/users/aot

# datasets
IMAGENET_CLUSTER_DIR=/gpfs/fs1/projects/ent_joc/datasets/image/imagenet
IMAGENET_DIR=/datasets/imagenet/

# scripts
WORKSPACE_MNT=/workspace_aot
EXP_DIR=${WORKSPACE_MNT}/exp/${EXP_NAME}
EXP_DIR_CLUSTER=${HOME_DIR}/exp/${EXP_NAME}
mkdir -p ${EXP_DIR_CLUSTER}

MOUNTS="--container-mounts=${IMAGENET_CLUSTER_DIR}:${IMAGENET_DIR},${HOME_DIR}:${WORKSPACE_MNT}"
GPU_ARGS="--gres=gpu:${SLURM_GPUS_PER_NODE} --gpus-per-node=${SLURM_GPUS_PER_NODE}"

export HYDRA_FULL_ERROR=1

# Copy Codebase from Github, Build dependencies
BRACH='main'
srun --mpi=pmix \
    --ntasks=$SLURM_JOB_NUM_NODES \
    --ntasks-per-node=1 \
    --container-image=nvcr.io/nvidia/pytorch:22.04-py3 \
    "${MOUNTS}" \
    ${GPU_ARGS} \
    --container-name=joc${SLURM_JOB_ID} \
    bash -c "git clone https://github.com/suiyoubi/multimodal.git; \
    cd /workspace/multimodal; \
    git fetch && git pull && git checkout ${BRANCH}; \
    pip install -e .; \
    cd /workspace/multimodal/examples/flava; \
    pip install -r requirements.txt "



# Run Training scripts based on the previously created container
OUTFILE="${EXP_DIR_CLUSTER}/slurm-${EXP_NAME}-%j-%n.out"
ERRFILE="${EXP_DIR_CLUSTER}/error-${EXP_NAME}-%j-%n.out"

srun -o ${OUTFILE} -e ${ERRFILE} ${GPU_ARGS} \
    --container-name=joc${SLURM_JOB_ID} "${MOUNTS}" \
    --ntasks-per-node=${SLURM_GPUS_PER_NODE} \
    -N ${SLURM_JOB_NUM_NODES} \
    bash -c "cd /workspace/multimodal/examples/flava; \
             SAVE_DIR=${EXP_DIR} IMAGENET_TAR=. python train.py \
                  config=configs/pretraining/debug.yaml \
                  training.lightning.devices=${SLURM_GPUS_PER_NODE} \
                  training.lightning.num_nodes=${SLURM_JOB_NUM_NODES}"

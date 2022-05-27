#!/bin/bash
#SBATCH -A wwfo_sae_misc
#SBATCH -p batch_short_dgx2h_m2 # batch, batch_16GB, Backfill... Check draco onboarding guide.
#SBATCH -N 2                  # number of nodes
#SBATCH -t 1:00:00              # wall time
#SBATCH -J "ml-model.flava"     # job name (<< CHANGE ! >>)
#SBATCH --exclusive             # exclusive node access
#SBATCH --mem=0                 # all mem avail
#SBATCH --mail-type=FAIL        # only send email on failure
#SBATCH --ntasks-per-node=16    # n tasks per machine
#SBATCH --overcommit
#SBATCH --gpus-per-node=16     # n gpus per machine <required>
#SBATCH --gres=gpu:16            # Specify the number of GPUs even if exclusiveNeeded for pytorch
#SBATCH --nv-meta="ml-model:flava"

set -euxo pipefail

EXP_NAME="profile-flava-multinode"
CONTAINER="gitlab-master.nvidia.com#wdykas/mm-containers:1.0"

HOME_DIR=/home/wdykas/ao-repo/multimodal

# datasets
IMAGENET_CLUSTER_DIR=/gpfs/fs1/projects/ent_joc/datasets/image/imagenet
IMAGENET_DIR=/datasets/imagenet/


# scripts
WORKSPACE_MNT=/workspace
EXP_DIR=${WORKSPACE_MNT}/exp/${EXP_NAME}
EXP_DIR_CLUSTER=${HOME_DIR}/exp/${EXP_NAME}

mkdir -p ${EXP_DIR_CLUSTER}


MOUNTS="--container-mounts=${IMAGENET_CLUSTER_DIR}:${IMAGENET_DIR},${HOME_DIR}:${WORKSPACE_MNT}"

export HYDRA_FULL_ERROR=1


OUTFILE="${EXP_DIR_CLUSTER}/slurm-${EXP_NAME}-%j-%n.out"
ERRFILE="${EXP_DIR_CLUSTER}/error-${EXP_NAME}-%j-%n.out"

srun -N 2 --container-name mm-container --container-image=gitlab-master.nvidia.com#wdykas/mm-containers:1.0 "${MOUNTS}" --gpus-per-node 16 bash -c "/workspace/install_req.sh"


srun -o ${OUTFILE} -e ${ERRFILE} --container-name mm-container "${MOUNTS}" \
  bash -c "SAVE_DIR=${EXP_DIR} IMAGENET_TAR=. nsys profile --delay=200 --duration=20 --nic-metrics=true --gpu-metrics-device all --trace=cuda,nvtx,osrt,cublas,cudnn python /workspace/examples/flava/train.py config=/workspace/examples/flava/configs/pretraining/debug.yaml training.lightning.devices=16 training.lightning.num_nodes=2"
# bash -c "python ptl-example.py"

#!/bin/bash

# usage: sbatch [-N nodes] [--partition mypartition] path/to/sparam-comms.sh

# see: https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-efa-using.html

#SBATCH --job-name=4Nodes_Highres_Train_Stage1_All_RES_GREC_GRES_TokenFormat_Epoch1
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --partition=learnai
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1

#SBATCH --output=./slurm_logs/4Nodes_Highres_Train_Stage1_All_RES_GREC_GRES_TokenFormat_Epoch1.out
#SBATCH --error=./slurm_logs/4Nodes_Highres_Train_Stage1_All_RES_GREC_GRES_TokenFormat_Epoch1.err

set -ex
echo "PARAM benchmark - Communication benchmarks: comms.py"

date

# edit to load conda env within which to test
source activate shikra2

export MODULEPATH=/opt/slurm/etc/files/modulesfiles/:$MODULEPATH
module load \
  cuda/11.6 \
  nccl/2.12.7-cuda.11.6 \
  nccl_efa/1.15.1-nccl.2.12.7-cuda.11.6

echo "PARAM benchmark - Communication benchmarks: comms.py"

date

export FI_EFA_USE_DEVICE_RDMA=1 # use for p4d
export FI_EFA_FORK_SAFE=1
export NCCL_PROTO=simple
export FI_LOG_LEVEL=1
export FI_PROVIDER=efa
export FI_EFA_ENABLE_SHM_TRANSFER=1
export NCCL_DEBUG=INFO

RDV_ADDR=$(hostname)
WORLD_SIZE=$SLURM_JOB_NUM_NODES

nproc_per_node=$SLURM_GPUS_ON_NODE
nnodes=$WORLD_SIZE

begin_size=1M  # --b
end_size=2G    # --e
num_iters=100  # --n
step_factor=2  # --f
blocking=1     # --z
collective=all_reduce


# in the output, should see lines like:
#
# ... <log EFA is engaged>
# NCCL INFO NET/OFI Selected Provider is efa (found 4 nics)

# ... <log Libfabric>
# NCCL INFO Using network AWS Libfabric

# ... <log aws-ofi-nccl>
# NCCL INFO NET/OFI Using aws-ofi-nccl 1.6.0 # 1.5.0 OK

# ... <log CUDA max version supported (not version used)>
# NCCL INFO cudaDriverVersion 12000

# ... <log send GDRDMA>
# [send] via NET/AWS Libfabric/1/GDRDMA

# ... <log recieved GDRRMA>
# [receive] via NET/AWS Libfabric/3/GDRDMA

# ... <log for 2 nodes, 1 GiB msg size, AlgBW =~ 42, BusBW =~ 77>
# 0: 	COMMS-RES-all_reduce-float32        1073741824         41.187      77.226

# if all log checks pass, and your runs do not perform well on inter-node communication,
# the problem is likely in your custom env and/or libs

srun torchrun \
--nproc_per_node $nproc_per_node \
--nnodes $nnodes \
--rdzv_id $SLURM_JOB_ID \
--rdzv_backend c10d \
--rdzv_endpoint $RDV_ADDR \
mllm/pipeline/finetune.py config/shikra_pretrain_concat8_stage1.py \
--cfg-options model_args.model_name_or_path=/fsx/spraman3/2023/Grounding_GPT4RoI/base_models/vicuna-7b-v1.3 \
--output_dir /fsx/spraman3/2023/shikra_gres_highres/exp/Train/4Nodes_Highres_Train_Stage1_All_RES_GREC_GRES_TokenFormat_Epoch1  \
--per_device_train_batch_size 8 --num_train_epochs 2
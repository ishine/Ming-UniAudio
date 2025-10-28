set -xv
set -eu

unset PythonPath
unset NCCL_DEBUG

export PYTHONPATH="./${PYTHONPATH:+:$PYTHONPATH}"


WORLD_SIZE=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
RANDOM_PORT=$[$RANDOM + 20000]
MASTER_PORT=${MASTER_PORT:-${RANDOM_PORT}}
NUM_PROCESSES=$(nvidia-smi -L | wc -l)

NUM_PROCESSES=$((NUM_PROCESSES*WORLD_SIZE))
#export CUDA_LAUNCH_BLOCKING=1

accelerate_config="$(dirname $1)/accelerate_config.yaml"
train_cmd="accelerate launch \
    --config_file ${accelerate_config} \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes $NUM_PROCESSES \
    --num_machines $WORLD_SIZE \
"

${train_cmd} sft/train.py $1

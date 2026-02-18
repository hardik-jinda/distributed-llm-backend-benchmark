#!/bin/bash

CONFIG_FILE=config_4.txt
SCRIPT=dsccl.py
MASTER_PORT=29501

# Parse config and build core list + rank count
CORE_LIST=""
NUM_RANKS=0

while read -r line; do
    [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue

    RANK=$(echo $line | awk '{print $1}')
    CORES=$(echo $line | awk '{print $2}')
    THREADS=$(echo $line | awk '{print $3}')

    if [ -z "$CORES" ]; then
        echo "Invalid config line: $line"
        exit 1
    fi

    if [ -z "$CORE_LIST" ]; then
        CORE_LIST=$CORES
    else
        CORE_LIST="$CORE_LIST,$CORES"
    fi

    NUM_RANKS=$((NUM_RANKS + 1))
done < "$CONFIG_FILE"

echo "Launching with $NUM_RANKS ranks"
echo "Core binding list: $CORE_LIST"
export CCL_LOG_LEVEL=debug

############################################
# oneCCL tuning variables (uncomment to test)
############################################

# export CCL_WORKER_COUNT=2
# Values: 1,2,4,... (number of communication worker threads per rank)

# export CCL_WORKER_AFFINITY=auto
# Values: auto | 0,1 | 4,5,6 (comma-separated core list)

# export CCL_WORKER_MEM_AFFINITY=auto
# Values: auto | 0 | 1 | 0,1 (NUMA node list)

export CCL_ALLREDUCE=direct
# Values: topo | direct | rabenseifner | nreduce | ring | double_tree | recursive_doubling | 2d

# export CCL_ALLGATHER=ring
# Values: topo | direct | naive | flat | multi_bcast | ring

# export CCL_FUSION=1
# Values: 0 (disable) | 1 (enable)

# export CCL_FUSION_BYTES_THRESHOLD=1048576
# Values: any integer (bytes), e.g., 65536 | 1048576 | 4194304

# export CCL_FUSION_COUNT_THRESHOLD=4
# Values: any integer, e.g., 2 | 4 | 8

# export CCL_FUSION_CYCLE_MS=1
# Values: integer (ms), e.g., 1 | 5 | 10

# export CCL_BLOCKING_WAIT=0
# Values: 0 (spin/progress) | 1 (blocking wait, lower CPU usage)

# export CCL_SPIN_COUNT=1000
# Values: integer, e.g., 100 | 1000 | 10000

# export CCL_ATL_TRANSPORT=mpi
# Values: mpi | ofi

# export CCL_ATL_SHM=0
# Values: 0 (disable) | 1 (enable OFI shared memory provider)

############################################

deepspeed \
    --num_gpus $NUM_RANKS \
    --bind_cores_to_rank \
    --bind_core_list $CORE_LIST \
    --master_port $MASTER_PORT \
    $SCRIPT

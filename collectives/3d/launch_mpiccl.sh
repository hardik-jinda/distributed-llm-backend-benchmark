#!/bin/bash

RANKS=8   
export MASTER_PORT=29501
# export I_MPI_COLL_EXTERNAL=1

export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=1
export I_MPI_PIN_PROCESSOR_LIST=16,17,18,19,20,21,22
export I_MPI_PIN_ORDER=compact

export I_MPI_DEBUG=10
export CCL_ATL_TRANSPORT=ofi
# oneCCL settings
export CCL_WORKER_COUNT=1
export CCL_WORKER_AFFINITY=8,9,10,11,12,13,14,15
export CCL_LOG_LEVEL=debug
export CCL_LOG_FILE=ccl_log_rank_%r.txt


echo "Launching MPI + oneCCL 3D Allreduce Benchmark..."
echo "Ranks: $RANKS"

mpirun -n ${RANKS} python mpitorchccl.py
#!/bin/bash

# Launch script for OpenMPI backend
# Usage: ./launch_openmpi.sh

# Configuration
CONFIG_FILE="config/baseline_config.yaml"
WORLD_SIZE=4
CORES_PER_RANK=14

# Set number of threads per rank
export OMP_NUM_THREADS=$CORES_PER_RANK
export MKL_NUM_THREADS=$CORES_PER_RANK

# Disable nested parallelism
export OMP_NESTED=FALSE

# Launch with OpenMPI
mpirun -np $WORLD_SIZE \
       --bind-to core \
       --map-by socket:PE=$CORES_PER_RANK \
       -x OMP_NUM_THREADS \
       -x MKL_NUM_THREADS \
       python run_mpi.py \
       --config $CONFIG_FILE \
       --backend openmpi

echo ""
echo "OpenMPI experiment completed!"

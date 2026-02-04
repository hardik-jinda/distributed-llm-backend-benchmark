#!/bin/bash

# Launch script for IntelMPI backend
# Usage: ./launch_intelmpi.sh

# Configuration
CONFIG_FILE="config/baseline_config.yaml"
WORLD_SIZE=4
CORES_PER_RANK=14

# Set number of threads per rank
export OMP_NUM_THREADS=$CORES_PER_RANK
export MKL_NUM_THREADS=$CORES_PER_RANK

# Disable nested parallelism
export OMP_NESTED=FALSE

# Intel MPI specific settings
export I_MPI_PIN_DOMAIN=omp
export I_MPI_FABRICS=shm
export I_MPI_SHM_LMT=shm

# Launch with Intel MPI
mpirun -np $WORLD_SIZE \
       -ppn $WORLD_SIZE \
       -genv OMP_NUM_THREADS=$CORES_PER_RANK \
       -genv MKL_NUM_THREADS=$CORES_PER_RANK \
       python run_mpi.py \
       --config $CONFIG_FILE \
       --backend intelmpi

echo ""
echo "IntelMPI experiment completed!"

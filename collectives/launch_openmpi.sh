#!/bin/bash

echo "Running threading comparison tests..."

export OMP_NUM_THREADS=2
mpirun -np 2 --report-bindings --map-by slot:PE=2 \
    python openmpi.py

mpirun -np 4 --report-bindings --map-by slot:PE=2 \
    python openmpi.py

mpirun -np 16 --report-bindings --map-by slot:PE=2 \
    python openmpi.py
#!/bin/bash

echo "Running threading comparison tests..."

export OMP_NUM_THREADS=2
mpirun -np 2 --map-by slot:PE=2 \
    python intelmpi.py

mpirun -np 4 --map-by slot:PE=2 \
    python intelmpi.py

mpirun -np 8 --map-by slot:PE=2 \
    python intelmpi.py

mpirun -np 16 --map-by slot:PE=2 \
    python intelmpi.py
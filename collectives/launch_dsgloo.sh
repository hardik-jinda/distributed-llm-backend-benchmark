#!/bin/bash

echo "Running threading comparison tests..."

export OMP_NUM_THREADS=2

deepspeed  --master_port 29501 --include localhost:0,1 dsgloo.py

export OMP_NUM_THREADS=2

deepspeed  --master_port 29501 --include localhost:0,1,2,3 dsgloo.py

export OMP_NUM_THREADS=2

deepspeed  --master_port 29501 --include localhost:0,1,2,3,4,5,6,7 dsgloo.py

export OMP_NUM_THREADS=2

deepspeed  --master_port 29501 --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 dsgloo.py

# Collectives experiment

This folder contains test script and results for single collective experiments.
We try to benchmark the follwing collective functions:

- Allreduce
- Allgather
- Broadcast
- Reduce
- Gather

To run experiments with OpenMPI:

```bash
mpirun -np <num_processes> python openmpi.py 
```
To run experiments with IntelMPI:

```bash
source /opt/intel/oneapi/setvars.sh
mpirun -np <num_processes> python intelmpi.py 
```
To run experiments with Deepspeed + gloo:

```bash
deepspeed  --master_port 29501 --include localhost:0,1 dsgloo.py
```

To run experiments with deepspeed + oneCCL:

```bash
source /opt/intel/oneapi/setvars.sh
deepspeed  --master_port 29501 --include localhost:0,1 dsccl.py
```

---

The `stats.py` file is used to calculate various statistics from the runs generated. To use it simply run:

```bash
python stats.py
```
Specify the folder where the result jsons are stored in the file.

It calculates various metrics such as:
- mean
- median
- standard deviation
- 95th percentile
- 99th percentile
- min
- max


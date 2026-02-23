    #!/bin/bash

    RANKS=8   
    export MASTER_PORT=29501
    export I_MPI_COLL_EXTERNAL=1

    export I_MPI_PIN=1
    export I_MPI_PIN_DOMAIN=1
    export I_MPI_PIN_PROCESSOR_LIST=20,21,22,23,24,25,26,27
    export I_MPI_PIN_ORDER=compact

    export I_MPI_DEBUG=10

    # oneCCL settings
    export CCL_WORKER_COUNT=2
    export CCL_WORKER_AFFINITY=4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
    export CCL_LOG_LEVEL=debug
    export CCL_LOG_FILE=ccl_log_rank_%r.txt


    echo "Launching MPI + oneCCL 3D Allreduce Benchmark..."
    echo "Ranks: $RANKS"

    mpirun -n ${RANKS} python mpitorchccl.py
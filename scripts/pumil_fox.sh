#!/bin/sh

DATASET="fox"
N=20

experiment () {
    prior=$1
    echo $DATASET $prior
    for i in $(seq 1 $N); do
        LOG=".results/pumil_${DATASET}_${prior}.log"
        (python pumil.py --dataset $DATASET --prior $prior >> $LOG) &
    done
    wait
}

experiment 0.1
experiment 0.2
experiment 0.3
experiment 0.4
experiment 0.5
experiment 0.6
experiment 0.7

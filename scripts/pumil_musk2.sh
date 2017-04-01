#!/bin/sh

DATASET="musk2"
N=50

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
experiment 0.4
experiment 0.7

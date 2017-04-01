#!/bin/sh

DATASET="elephant"
N=50

experiment () {
    prior=$1
    for i in $(seq 1 $N); do
        echo $DATASET $prior $i
        LOG=".results/puskc_${DATASET}_${prior}.log"
        (python puskc.py --dataset $DATASET --prior $prior >> $LOG)
    done
}

experiment 0.1
experiment 0.4
experiment 0.7

#!/bin/sh

DATASET="musk1"
MODEL="dsdd"
N=20

experiment () {
    prior=$1
    for i in $(seq 1 $N); do
        echo $DATASET $prior $i
        LOG=".results/${MODEL}_${DATASET}_${prior}.log"
        (python ${MODEL}.py --dataset $DATASET --prior $prior >> $LOG)
    done
}

experiment 0.1
experiment 0.2
experiment 0.3
experiment 0.4
experiment 0.5
experiment 0.6
experiment 0.7

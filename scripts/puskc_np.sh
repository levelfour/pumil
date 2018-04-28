#!/bin/sh

DATASET="$1"
N=24
N_PROC=32
prior=0.7
HOSTNAME="$(hostname)"

run () {
    NP=$1
    LOG=".results/puskc_${DATASET}_${prior}_${NP}.log"
    printf ${NP}'\n%.0s' $(seq 1 $N) | xargs -t -P$N_PROC -n1 python puskc.py --dataset $DATASET --prior $prior --np >> $LOG
}

if [ $HOSTNAME == "b1" ]; then
    run 20
elif [ $HOSTNAME == "b2" ]; then
    run 120
elif [ $HOSTNAME == "b5" ]; then
    run 140
elif [ $HOSTNAME == "b6" ]; then
    run 160
fi

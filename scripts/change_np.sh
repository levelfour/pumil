#!/bin/sh

METHOD="$1"
DATASET="$2"
N=24
N_PROC=32
prior=0.7
HOSTNAME="$(hostname)"

run () {
    NP=$1
    LOG=".results-np/${METHOD}_${DATASET}_${prior}_${NP}.log"
    printf ${NP}'\n%.0s' $(seq 1 $N) | xargs -t -P$N_PROC -n1 python ${METHOD}.py --dataset $DATASET --prior $prior --nu 180 --np >> $LOG
}

run $3

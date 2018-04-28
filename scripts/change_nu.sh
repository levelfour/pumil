#!/bin/sh

METHOD="$1"
DATASET="$2"
N=24
N_PROC=32
prior=0.7
HOSTNAME="$(hostname)"

run () {
    NU=$1
    LOG=".results-nu/${METHOD}_${DATASET}_${prior}_${NU}.log"
    printf ${NU}'\n%.0s' $(seq 1 $N) | xargs -t -P$N_PROC -n1 python ${METHOD}.py --dataset $DATASET --prior $prior --np 20 --nu >> $LOG
}

run $3

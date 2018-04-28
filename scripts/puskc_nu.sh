#!/bin/sh

DATASET="$1"
N=24
N_PROC=32
prior=0.7
HOSTNAME="$(hostname)"

run () {
    NU=$1
    LOG=".results-nu/puskc_${DATASET}_${prior}_${NU}.log"
    printf ${NU}'\n%.0s' $(seq 1 $N) | xargs -t -P$N_PROC -n1 python puskc.py --dataset $DATASET --prior $prior --nu >> $LOG
}

if [ $HOSTNAME == "b1" ]; then
    run 240
elif [ $HOSTNAME == "b2" ]; then
    run 300
elif [ $HOSTNAME == "b5" ]; then
    run 360
elif [ $HOSTNAME == "b6" ]; then
    run 420
fi
